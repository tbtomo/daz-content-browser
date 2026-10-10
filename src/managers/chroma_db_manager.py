""" Manages interactions with ChromaDB for vector storage and retrieval. """

import chromadb
import logging
import math
import os
import re
from typing import List, Optional
from embedding_utils import generate_embeddings
from query_translation import contains_japanese, translate_query, vocabulary_of

logger = logging.getLogger(__name__)


def and_term_fraction() -> float:
    """How near each term of a multi-term search a product must be, as the fraction
    of the catalogue it must rank within for every term (AND_TERM_FRACTION, default
    0.25). Measured on 4,601 products: 0.10 dropped good matches whose phrasing
    differs from the term (chairs for '椅子 木製'), 0.25 kept the phrase search's
    top hits while cutting the list to the few hundred near every term."""
    try:
        value = float(os.getenv("AND_TERM_FRACTION", "0.25"))
    except ValueError:
        return 0.25
    return min(max(value, 0.01), 1.0)

# ASCII, curly and full-width double quotes all delimit a phrase, so a query typed
# with a Japanese IME behaves the same as one typed on an English keyboard.
_QUOTES = '"“”＂'
_TERM_RE = re.compile(rf'[{_QUOTES}]([^{_QUOTES}]+)[{_QUOTES}]|(\S+)')


def split_search_terms(prompt: str) -> List[str]:
    """Splits a search query into the terms that must all match.

    Whitespace (including the full-width space) separates terms; double quotes keep a
    phrase together, so ``"long hair" red`` is two terms. A stray unmatched quote is
    dropped rather than becoming part of a term. Repeated terms count once.
    """
    terms = []
    for quoted, bare in _TERM_RE.findall(prompt or ""):
        term = (quoted or bare.strip(_QUOTES)).strip()
        if term:
            terms.append(term)
    return list(dict.fromkeys(terms))


def join_search_terms(terms: List[str]) -> str:
    """Inverse of split_search_terms: multi-word terms are quoted so the result,
    typed back in as a query, means the same thing."""
    return " ".join(f'"{t}"' if len(t.split()) > 1 else t for t in terms)


def build_where_clause(
    tags: Optional[List[str]] = None,
    artists: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    compatible_figures: Optional[List[str]] = None,
):
    """
    Dynamically builds a ChromaDB 'where' filter for multiple fields.
    Uses $and for criteria across different fields (e.g., category AND artist)
    and $or for criteria within the same field (e.g., category is Clothing OR Hair).
    """

    # Conditions that must ALL be met are added to this list
    if not any([tags, artists, categories, compatible_figures]):
        return None

    and_conditions = []

    # Helper function to create a filter block for a list of values
    def create_or_condition(field_name: str, values: List[str]):
        if not values:
            return

        # For the 'category' field, which is a single string, we use exact match ($eq)
        if field_name == "category":
            conditions = [{field_name: {"$eq": value}} for value in values]
        # For fields stored as JSON strings of lists, we use substring matching ($contains)
        else:
            conditions = [{field_name: {"$contains": value}} for value in values]

        if len(conditions) > 1:
            and_conditions.append({"$or": conditions})
        elif conditions:
            and_conditions.append(conditions[0])

    # Build conditions for each filter type passed to the function
    if tags:               create_or_condition("tags", tags)
    if artists:            create_or_condition("artist", artists)
    if categories:         create_or_condition("category", categories)
    if compatible_figures: create_or_condition("compatible_figures", compatible_figures)

    # Return the final filter structure for the ChromaDB query
    if not and_conditions:
        return None
    if len(and_conditions) == 1:
        return and_conditions[0]

    return {"$and": and_conditions}


class ChromaDbManager:
    """Handles all interactions with ChromaDB."""

    def __init__(self, chroma_db_path:str, collection_name:str):
        """Create a ChromaDB manager instance.

        Args:
            chroma_db_path (str): Path to the ChromaDB database directory.
            collection_name (str): Name of the collection to use within ChromaDB.
        """
        
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Example for 768-dim embeddings
        )
        self._vocabulary = None
        self._vocabulary_count = None

        logger.info(f"ChromaDB opened — path: {self.chroma_db_path!r}, collection: {self.collection_name!r}")
        
        
    def _clean_metadata(self, item: dict) -> dict:
        """ Cleans metadata dictionary by removing None values and non-serializable types.

        Args:
            item (dict): Input metadata dictionary.

        Returns:
            dict: Cleaned metadata dictionary.
        """
        clean = {}
        for key, value in item.items():
            if value is not None and isinstance(value, (str, int, float, bool)):
                clean[key] = value
        return clean
    
    def reconnect(self) -> None:
        """Recreates the ChromaDB client and refreshes the collection reference.

        Call this when an upsert fails with a stale-connection or missing-collection
        error so the next operation gets a fresh handle.
        """
        self.client = chromadb.PersistentClient(path=self.chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB reconnected — path: {self.chroma_db_path!r}, collection: {self.collection_name!r}")

    def get_all_ids(self) -> set:
        """Returns the set of all document IDs currently stored in the collection."""
        return set(self.collection.get(include=[])["ids"])

    def reset_collection(self) -> None:
        """Deletes and recreates the default collection, leaving it empty and ready for a fresh index.

        Updates self.collection so existing references stay valid.
        """
        logger.info(f"Resetting ChromaDB collection {self.collection_name!r}...")
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            logger.warning(f"Collection {self.collection_name!r} did not exist; nothing to delete.")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection {self.collection_name!r} reset and ready.")

    def search(
        self,
        prompt: str,
        tags: Optional[List[str]] = None,
        artists: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        compatible_figures: Optional[List[str]] = None,
        limit: int = 10,
        offset: int = 0,
        max_results: int = 500,
        score_threshold: float = 1.0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ):
        """Searches the ChromaDB collection with the given parameters.

        A prompt of several whitespace-separated terms is an AND search. Results are
        ranked, and scored, by the terms embedded together as one phrase — exactly as a
        single-term search is — and a product is kept only if it is also among the
        nearest ``AND_TERM_FRACTION`` of the catalogue for *each* term on its own.
        Double quotes keep a phrase together as one term.

        Why not score by the weakest term: single words embed poorly against these
        short product texts ('red' alone is nearest to RuDy_Studio material packs),
        so the weakest-term score ranked worse than the phrase on every query tried.
        The phrase ranks well; the per-term cut is what makes it an AND. A rank
        cut rather than a score cut, because each word's scores sit in its own band.

        Args:
            prompt (str): The search prompt to generate the query embedding(s).
            tags (List[str], optional): List of tags to filter by. Defaults to None.
            artists (List[str], optional): List of artists to filter by. Defaults to None.
            categories (List[str], optional): List of categories to filter by. Defaults to None.
            compatible_figures (List[str], optional): List of compatible figures to filter by. Defaults to None.
            limit (int, optional): Number of results to return from the filtered set. Defaults to 10.
                Used by the MCP /query endpoint. The UI /search endpoint uses max_results instead.
            offset (int, optional): Offset into the filtered result set. Defaults to 0.
                Used by the MCP /query endpoint. The UI /search endpoint always passes 0.
            max_results (int, optional): Maximum number of candidates to fetch from ChromaDB.
                This is the fixed pool size — independent of limit/offset — so total_hits is
                stable across requests. Defaults to 500. The UI /search endpoint controls this
                directly; the MCP /query endpoint leaves it at the default.
            score_threshold (float, optional): Maximum cosine distance for a result to be
                included. Defaults to 1.0 (include everything).
            sort_by (str, optional): Field to sort by. Use 'relevance' for cosine-distance order
                (default), or any metadata field name (e.g. 'name', 'artist').
            sort_order (str, optional): 'ascending' or 'descending'. Only applied when
                sort_by is not 'relevance'. Defaults to 'descending'.

        Returns:
            dict: Search results including total_hits, limit, offset, and list of results.
        """

        # --- 1. Split, Translate, then Generate Query Embeddings ---
        # The index is English. A Japanese term is translated first so the embedding
        # model compares like with like; anything else passes through untouched.
        # Terms are translated one by one, which also lets each hit the glossary,
        # whose entries only match a whole query. The catalogue's vocabulary lets the
        # translator catch a term it only romanised (see translate_query).
        terms = split_search_terms(prompt) or [prompt]
        known_words = self._catalog_words() if contains_japanese(prompt) else None
        embedded_terms, any_translated = [], False
        for term in terms:
            text, source = translate_query(term, known_words)
            embedded_terms.append(text)
            any_translated = any_translated or source is not None
        phrase = " ".join(embedded_terms)
        texts = [phrase] + (embedded_terms if len(terms) > 1 else [])
        query_embeddings = generate_embeddings(texts, is_query=True)

        # --- 2. Build the Combined Metadata Filter ---
        where_filter = build_where_clause(
            tags=tags,
            artists=artists,
            categories=categories,
            compatible_figures=compatible_figures,
        )
        if where_filter:
            logger.debug(f"Applying metadata filter: {where_filter}")

        # Fixed candidate pool — does not grow with offset/limit so total_hits is stable.
        query_limit = max_results

        # --- 3. Query ChromaDB ---

        results = self.collection.query (
            query_embeddings=query_embeddings[:1].tolist(),
            n_results=query_limit,
            where=where_filter,
            include=["metadatas", "distances"],
        )
        ranked = list(zip(results["ids"][0], results["distances"][0], results["metadatas"][0])) \
            if results["ids"] else []

        # --- 4. AND: Keep Products Near Every Term ---
        if len(terms) > 1:
            in_scope = (len(self.collection.get(where=where_filter, include=[])["ids"])
                        if where_filter else self.collection.count())
            per_term = max(1, math.ceil(in_scope * and_term_fraction()))
            term_results = self.collection.query(
                query_embeddings=query_embeddings[1:].tolist(),
                n_results=per_term,
                where=where_filter,
                include=[],
            )
            for ids in term_results["ids"]:
                near = set(ids)
                ranked = [r for r in ranked if r[0] in near]

        # --- 5. Post-process Results (Filtering by Score) ---
        processed_results = [
            {
                "id": sku,
                "distance": dist,
                "relevance_score": round(1.0 - dist, 4),
                "metadata": meta,
            }
            for sku, dist, meta in ranked
            if dist <= score_threshold
        ]

        # --- 6. Sorting Logic ---
        reverse_order = sort_order == "descending"
        if sort_by != "relevance":
            # Sort by a metadata field, handling potential missing keys gracefully
            processed_results.sort(
                key=lambda x: x["metadata"].get(sort_by) or "",  # Fallback for sorting
                reverse=reverse_order,
            )
        # Note: ChromaDB already returns results sorted by relevance (distance ascending)

        # --- 7. Apply Pagination and Return ---
        paginated_results = processed_results[offset : offset + limit]

        logger.debug(f"Result set size: {len(paginated_results)}")

        return {
            "total_hits": len(processed_results),
            "limit": limit,
            "offset": offset,
            "results": paginated_results,
            # Present only when the query was rewritten, so a caller can show what
            # was actually searched rather than what the user typed.
            "translated_from": prompt if any_translated else None,
            # What was embedded, quoted so it reads back as the same AND query.
            "query": join_search_terms(embedded_terms) if len(terms) > 1 else embedded_terms[0],
            "terms": embedded_terms,
        }

    def _catalog_words(self) -> set:
        """Every word the indexed product texts use, rebuilt when the index changes."""
        count = self.collection.count()
        if self._vocabulary is None or self._vocabulary_count != count:
            documents = self.collection.get(include=["documents"])["documents"]
            self._vocabulary = vocabulary_of(documents)
            self._vocabulary_count = count
        return self._vocabulary
        

    def get_db_stats(self):
        """Returns the total document count from the ChromaDB collection.

        Returns:
            dict: A dictionary containing total document count.
        """
        return {"total_docs": self.collection.count()}

