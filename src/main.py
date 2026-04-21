from src.qa_pipeline import answer_question

def main():
    print("🚀 Document-Aware Q&A System")

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = answer_question(query)

        print("\n💡 Answer:")
        print(answer)


if __name__ == "__main__":
    main()