# Chunking Strategy Notes

RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
then sentences, then words -- only falling back to a hard character cut
if nothing else fits within chunk_size.

Typical starting point for a personal knowledge base: chunk_size=500,
chunk_overlap=50. Overlap keeps context from being severed mid-idea across
chunk boundaries, which matters a lot for retrieval quality.

Tune this empirically: too small and answers lose context, too large and
retrieval gets noisy (irrelevant text riding along with the relevant part).
