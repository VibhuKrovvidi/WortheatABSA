import aspect_based_sentiment_analysis as absa
import transformers


text = ("We are great fans of Slack, but we wish the subscriptions "
        "were more accessible to small startups.")

name = 'absa/classifier-rest-0.2'
model = absa.BertABSClassifier.from_pretrained(name)
tokenizer = transformers.BertTokenizer.from_pretrained(name)
professor = absa.Professor()     # Explained in detail later on.
# text_splitter = absa.sentencizer()  # The English CNN model from SpaCy.
nlp = absa.Pipeline(model, tokenizer, professor)

# Break down the pipeline `call` method.
task = nlp.preprocess(text=text)
tokenized_examples = nlp.tokenize(task.examples)
input_batch = nlp.encode(tokenized_examples)
output_batch = nlp.predict(input_batch)
predictions = nlp.review(tokenized_examples, output_batch)
completed_task = nlp.postprocess(task, predictions)