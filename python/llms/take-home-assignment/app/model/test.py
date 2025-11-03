from rag_pipeline import RagPipeline
rp = RagPipeline()

print("✅ Number of context chunks:", len(rp.context_sections))
print("Sample chunk preview:", rp.context_sections[0][:100])
print("Test query result:", rp.query("How do I request a new credit card?"))
