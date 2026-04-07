from raptor import RetrievalAugmentation

with open("demo/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

ra = RetrievalAugmentation()
ra.add_documents(text)

# question = "How did Cinderella reach her happy ending?"
# context, layer_info = ra.retrieve(question, return_layer_information=True)
# answer = ra.answer_question(question)


# ra.save("demo/test_tree.pkl")
# ra2 = RetrievalAugmentation(tree="demo/test_tree.pkl")
# print(ra2.answer_question("How did Cinderella reach her happy ending?"))


# print("Context preview:", context[:500])
# print("Layer info sample:", layer_info[:5])
# print("Answer:", answer)

payload = ra.answer_question_with_metadata("How did Cinderella reach her happy ending?")
print(payload.keys())
print(payload["retrieval"].keys())
print(payload["retrieval"]["retrieved_nodes"][0].keys())
print(payload["answer"])
