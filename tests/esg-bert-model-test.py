from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()  # loading HF_TOKEN

model = AutoModelForSequenceClassification.from_pretrained("nbroad/ESG-BERT")
# print(model.config.id2label)


# testing outputs
classifier = pipeline("text-classification", model="nbroad/ESG-BERT")

samples = [
    "We reduced greenhouse gas emissions by 15%.",
    "Employee injury rates decreased by 10%.",
    "The company has 12 manufacturing plants.",
    "Revenue increased 20% year-over-year.",
    "Our board consists of 8 directors.",
    "We expanded renewable energy usage.",
    "The headquarters are located in Texas.",
    "Data privacy remains a key priority.",
]

for text in samples:
    result = classifier(text)[0]
    print(f"{result['label']:40} " f"{result['score']:.3f} " f"{text}")


"""
First output:
TEXT: The company reduced greenhouse gas emissions by 15%.
[{'label': 'GHG_Emissions', 'score': 0.9655773639678955}]

TEXT: We improved workplace safety and reduced injuries.
[{'label': 'Employee_Health_And_Safety', 'score': 0.9829734563827515}]

TEXT: The company headquarters are located in Texas.
[{'label': 'Business_Model_Resilience', 'score': 0.21386854350566864}]

TEXT: The annual shareholder meeting was held on May 10.
[{'label': 'Director_Removal', 'score': 0.36023256182670593}]

Next, testing on a larger sample:

"""
