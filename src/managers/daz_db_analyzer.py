import logging
import os
import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Products installed outside the DAZ store (Renderosity, manual ZIP installs, ...) are
# registered in the CMS but carry an empty token. They show up in DAZ Studio's Content
# Library but not in Smart Content. Give them a synthetic, stable SKU derived from the
# CMS row id so the rest of the pipeline — which is keyed on SKU — can carry them.
SKU_EXPR = "COALESCE(NULLIF(p.token, ''), 'cms-' || p.id)"

# Prefix used by the synthetic SKUs above. Callers use it to decide whether a product
# has a daz3d.com store page.
LOCAL_SKU_PREFIX = "cms-"


class DazDBAnalyzer:
    """Manages connections and queries to a PostgreSQL database for DAZ 3D content.
    """

    """SQL query components for fetching product data."""
    QUERY_BODY = f"""
        SELECT
            p.id AS product_id,
            p.name AS product_name,
            p.artists,
            {SKU_EXPR} AS sku,
            (p.token IS NULL OR p.token = '') AS is_local,
            COALESCE(p.last_update, p.date_installed) AS last_modified_date,
            REGEXP_REPLACE(
                STRING_AGG(DISTINCT final_compat.compatibility_name, ', '),
                '^, *| *, $', ''
            ) AS product_compatibility,
            COUNT(DISTINCT c.id) AS content_item_count,
            STRING_AGG(DISTINCT cat."fldCategoryName", ', ') AS categories,
            STRING_AGG(DISTINCT ct."fldType", ', ') AS content_types
        FROM dzcontent.product AS p
        LEFT JOIN dzcontent.content AS c ON p.id = c.product_id
        LEFT JOIN (
            SELECT
                c_inner.id AS content_id,
                COALESCE(
                    cb."fldCompatibilityBase",
                    CASE
                        WHEN c_inner.path ILIKE '%%/Genesis 9/%%' OR c_inner.filename ILIKE '%%Genesis 9%%' OR c_inner.filename ILIKE '%%G9%%' THEN 'Genesis 9'
                        WHEN c_inner.path ILIKE '%%/Genesis 8.1/%%' OR c_inner.filename ILIKE '%%Genesis 8.1%%' OR c_inner.filename ILIKE '%%G8_1%%' THEN 'Genesis 8.1'
                        WHEN c_inner.path ILIKE '%%/Genesis 8/%%' OR c_inner.filename ILIKE '%%Genesis 8%%' OR c_inner.filename ILIKE '%%G8%%' THEN 'Genesis 8'
                        WHEN c_inner.path ILIKE '%%/Genesis 3/%%' OR c_inner.filename ILIKE '%%Genesis 3%%' OR c_inner.filename ILIKE '%%G3%%' THEN 'Genesis 3'
                        WHEN c_inner.path ILIKE '%%/Genesis 2/%%' OR c_inner.filename ILIKE '%%Genesis 2%%' OR c_inner.filename ILIKE '%%G2%%' THEN 'Genesis 2'
                        ELSE NULL
                    END
                ) AS compatibility_name
            FROM dzcontent.content AS c_inner
            LEFT JOIN dzcontent.compatibility_base_content AS cbc ON c_inner.id = cbc.content_id
            LEFT JOIN dzcontent."tblCompatibilityBase" AS cb ON cbc.compatibility_base_id = cb."RecID"
        ) AS final_compat ON c.id = final_compat.content_id
        LEFT JOIN dzcontent.category_content AS cc ON c.id = cc.content_id
        LEFT JOIN dzcontent."tblCategories" AS cat ON cc.category_id = cat."RecID"
        LEFT JOIN dzcontent."tblType" AS ct ON c.content_type_id = ct."RecID"
    """

    """SQL query grouping and ordering clause."""
    QUERY_GROUPING = "GROUP BY p.id, p.name, p.artists, p.token, p.last_update, p.date_installed ORDER BY p.name"


    def __init__(self):        
        """Initializes the DazDBAnalyzer with database configuration from environment variables.
        Expects the following environment variables to be set:
            - DB_NAME
            - DB_USER
            - DB_HOST
            - DB_PORT
            - DB_PASS  (optional — DAZ Studio's default dzcms account has no password)
            - BATCH_SIZE (optional, defaults to 512)
        """
        required = ("DB_NAME", "DB_USER", "DB_HOST", "DB_PORT")
        missing = [k for k in required if not os.environ.get(k)]
        if missing:
            raise EnvironmentError(
                f"DazDBAnalyzer: missing required environment variable(s): {', '.join(missing)}"
            )

        db_config = {
            "dbname":   os.environ["DB_NAME"],
            "user":     os.environ["DB_USER"],
            "password": os.environ.get("DB_PASS", ""),
            "host":     os.environ["DB_HOST"],
            "port":     os.environ["DB_PORT"],
        }
        pool_max = int(os.getenv("DB_POOL_MAX", "5"))
        self._pool = psycopg2.pool.ThreadedConnectionPool(1, pool_max, **db_config)
        self.batch_size = int(os.getenv("BATCH_SIZE", 512))

    def _execute_query(self, sql, params=None):
        """Executes a SQL query and returns the results as a list of dictionaries.

        Args:
            sql (str): The SQL query to execute.    
            params (tuple, optional): Parameters to pass to the SQL query. Defaults to None.

        Returns:
            list: A list of dictionaries representing the query results.
        """

        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(sql, params)
                results = cur.fetchall() if cur.description else []
            return [dict(row) for row in results]
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error: {e}")
            return None
        finally:
            self._pool.putconn(conn)

    def get_all_skus(self):
        """Efficiently fetches a list of all product SKUs from PostgreSQL.

        Products without a DAZ store token get the synthetic 'cms-<id>' SKU defined
        by SKU_EXPR, so Content-Library-only products are included.

        Returns:
            list: A list of all SKUs in the database as strings.
        """
        logger.info("Fetching all valid SKUs from PostgreSQL...")
        sql = f"SELECT DISTINCT {SKU_EXPR} AS sku FROM dzcontent.product AS p"
        results = self._execute_query(sql)
        return [row['sku'] for row in results] if results else []

    def count_skus(self) -> int:
        """Returns the count of distinct SKUs in PostgreSQL.

        Uses DISTINCT to match the deduplication applied by get_all_skus(), since
        multiple product rows can share the same token.

        Returns:
            int: Number of distinct SKUs, or -1 on error.
        """
        sql = f"SELECT COUNT(DISTINCT {SKU_EXPR}) AS n FROM dzcontent.product AS p"
        results = self._execute_query(sql)
        if results is None:
            return -1
        return results[0]['n'] if results else 0

    def get_content_roots(self) -> list:
        """Returns all content root directories from the DAZ CMS database.

        Returns:
            list: A list of absolute path strings (e.g. ['X:/DAZ Libraries/Project', ...]).
        """
        sql = 'SELECT "fldBasePath" FROM dzcontent."tblBasePath" ORDER BY "RecID"'
        results = self._execute_query(sql)
        return [r['fldBasePath'] for r in results] if results else []

    def get_all_content_paths(self) -> list:
        """Returns (path, filename) for every file the CMS tracks.

        Used by the filesystem scanner to subtract already-indexed content from a
        walk of the content roots. Streamed with a server-side cursor because this
        table runs to hundreds of thousands of rows.

        Returns:
            list[tuple[str, str]]: (relative_path, filename) pairs.
        """
        conn = self._pool.getconn()
        try:
            with conn.cursor(name="content_paths") as cur:
                cur.itersize = 50000
                cur.execute("SELECT path, filename FROM dzcontent.content")
                return [(row[0], row[1]) for row in cur]
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error streaming content paths: {e}")
            return []
        finally:
            self._pool.putconn(conn)

    def get_asset_files_by_sku(self, sku: str) -> list:
        """Returns all asset files for a given product SKU from the DAZ CMS database.

        Each result contains the relative path within the content directory,
        the filename, and the content type label. The caller is responsible for
        combining path + filename with a content root to get an absolute disk path.

        Args:
            sku (str): The product SKU (token) to look up.

        Returns:
            list: A list of dicts with keys: path, filename, content_type.
                  Returns an empty list if the SKU is not found.
        """
        # Match on an indexable column rather than SKU_EXPR — a functional match would
        # force a sequential scan of dzcontent.content on every product detail view.
        if sku.startswith(LOCAL_SKU_PREFIX):
            try:
                predicate, value = "p.id = %s", int(sku[len(LOCAL_SKU_PREFIX):])
            except ValueError:
                logger.warning(f"Malformed local SKU {sku!r}.")
                return []
        else:
            predicate, value = "p.token = %s", sku

        sql = f"""
            SELECT
                c.path,
                c.filename,
                ct."fldType" AS content_type
            FROM dzcontent.content AS c
            JOIN dzcontent.product AS p ON p.id = c.product_id
            LEFT JOIN dzcontent."tblType" AS ct ON c.content_type_id = ct."RecID"
            WHERE {predicate}
              AND c.filename IS NOT NULL
              AND c.filename != ''
            ORDER BY c.path, c.filename
        """
        results = self._execute_query(sql, (value,))
        return results if results is not None else []

    def get_candidate_thumbnail_files(self, skus: list, per_sku: int = 12) -> dict:
        """Returns a few representative asset files per SKU, for thumbnail discovery.

        Products without a store page have no scraped image, so the ETL falls back to
        the companion PNG DAZ Studio writes next to an asset file. Fetching a handful
        of candidates per product in one query keeps this off the per-product path —
        SKU_EXPR is not indexable, so each query costs a scan of dzcontent.content.

        Args:
            skus (list): SKUs to look up.
            per_sku (int): Maximum candidate files to return per SKU.

        Returns:
            dict: {sku: [{'path': ..., 'filename': ...}, ...]}, .duf files first.
        """
        if not skus:
            return {}

        candidates = {}
        for i in range(0, len(skus), self.batch_size):
            sku_batch = skus[i:i + self.batch_size]
            placeholders = ','.join(['%s'] * len(sku_batch))
            sql = f"""
                SELECT sku, path, filename FROM (
                    SELECT
                        {SKU_EXPR} AS sku,
                        c.path,
                        c.filename,
                        ROW_NUMBER() OVER (
                            PARTITION BY {SKU_EXPR}
                            ORDER BY
                                CASE WHEN lower(c.filename) LIKE '%%.duf' THEN 0 ELSE 1 END,
                                c.path, c.filename
                        ) AS rn
                    FROM dzcontent.content AS c
                    JOIN dzcontent.product AS p ON p.id = c.product_id
                    WHERE {SKU_EXPR} IN ({placeholders})
                      AND c.filename IS NOT NULL
                      AND c.filename != ''
                ) AS ranked
                WHERE rn <= %s
            """
            rows = self._execute_query(sql, tuple(sku_batch) + (per_sku,))
            for row in rows or []:
                candidates.setdefault(row['sku'], []).append(
                    {'path': row['path'], 'filename': row['filename']}
                )
        return candidates

    def get_products_by_sku_list(self, skus):
        """Fetches full product data for a given list of SKUs, handling batching.
        
        Args:
            skus (list): List of SKUs to fetch data for.

        Returns:
            list: A list of dictionaries containing product data for the given SKUs.
        """
        if not skus:
            return []
        logger.info(f"Fetching full data for {len(skus)} products from PostgreSQL...")
        all_products = []
        for i in range(0, len(skus), self.batch_size):
            sku_batch = skus[i:i + self.batch_size]
            placeholders = ','.join(['%s'] * len(sku_batch))
            where_clause = f"WHERE {SKU_EXPR} IN ({placeholders})"
            sql = f"{self.QUERY_BODY} {where_clause} {self.QUERY_GROUPING}"
            batch_results = self._execute_query(sql, tuple(sku_batch))
            if batch_results:
                all_products.extend(batch_results)
        return all_products

    def close(self):
        """Closes all connections in the pool."""
        self._pool.closeall()