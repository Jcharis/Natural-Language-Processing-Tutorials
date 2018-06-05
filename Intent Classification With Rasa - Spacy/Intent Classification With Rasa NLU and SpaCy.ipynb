{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Intent Classification with Rasa NLU and SpaCy \n",
    "+ + A Libary for intent recognition and entity extraction based on SpaCy and Sklearn\n",
    "\n",
    "##### NLP = NLU+NLG+ More\n",
    "+ NLP = understand,process,interprete everyday human language\n",
    "+ NLU = unstructured inputs and convert them into a structured form that a machine can understand and act upon\n",
    "\n",
    "#### Uses\n",
    "+ Chatbot task\n",
    "+ NL understanding\n",
    "+ Intent classification\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![alt text](nlu_nlp_explain.png \"Title\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Installation\n",
    "+ pip install rasa_nlu\n",
    "+ python -m rasa_nlu.server &\n",
    "+ sklearn_crfsuite\n",
    "\n",
    "#### using spacy as backend\n",
    "+ pip install rasa_nlu[spacy]\n",
    "+ python -m spacy download en_core_web_md\n",
    "+ python -m spacy link en_core_web_md en\n",
    "  \n",
    "  = = Dataset = =\n",
    "+ demo-rasa.json\n",
    "+ config_spacy.yaml"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the Packages\n",
    "from rasa_nlu.training_data  import load_data\n",
    "from rasa_nlu.config import RasaNLUModelConfig\n",
    "from rasa_nlu.model import Trainer\n",
    "from rasa_nlu import config"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\n",
      "  \"rasa_nlu_data\": {\n",
      "    \"regex_features\": [\n",
      "      {\n",
      "        \"name\": \"zipcode\",\n",
      "        \"pattern\": \"[0-9]{5}\"\n",
      "      },\n",
      "      {\n",
      "        \"name\": \"greet\",\n",
      "        \"pattern\": \"hey[^\\\\s]*\"\n",
      "      }\n",
      "    ],\n",
      "    \"entity_synonyms\": [\n",
      "      {\n",
      "        \"value\": \"chinese\",\n",
      "        \"synonyms\": [\"Chinese\", \"Chines\", \"chines\"]\n",
      "      },\n",
      "      {\n",
      "        \"value\": \"vegetarian\",\n",
      "        \"synonyms\": [\"veggie\", \"vegg\"]\n",
      "      }\n",
      "    ],\n",
      "    \"common_examples\": [\n",
      "      {\n",
      "        \"text\": \"hey\", \n",
      "        \"intent\": \"greet\", \n",
      "        \"entities\": []\n",
      "      }, \n",
      "      {\n",
      "        \"text\": \"howdy\", \n",
      "        \"intent\": \"greet\", \n",
      "        \"entities\": []\n",
      "      }, \n",
      "      {\n",
      "        \"text\": \"hey there\",\n",
      "        \"intent\": \"greet\", \n",
      "        \"entities\": []\n",
      "      }, \n",
      "      {\n",
      "        \"text\": \"hello\", \n",
      "        \"intent\": \"greet\", \n",
      "        \"entities\": []\n",
      "      }, \n",
      "      {\n",
      "        \"text\": \"hi\", \n",
      "        \"intent\": \"greet\", \n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"good morning\",\n",
      "        \"intent\": \"greet\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"good evening\",\n",
      "        \"intent\": \"greet\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"dear sir\",\n",
      "        \"intent\": \"greet\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"yes\", \n",
      "        \"intent\": \"affirm\", \n",
      "        \"entities\": []\n",
      "      }, \n",
      "      {\n",
      "        \"text\": \"yep\", \n",
      "        \"intent\": \"affirm\", \n",
      "        \"entities\": []\n",
      "      }, \n",
      "      {\n",
      "        \"text\": \"yeah\", \n",
      "        \"intent\": \"affirm\", \n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"indeed\",\n",
      "        \"intent\": \"affirm\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"that's right\",\n",
      "        \"intent\": \"affirm\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"ok\",\n",
      "        \"intent\": \"affirm\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"great\",\n",
      "        \"intent\": \"affirm\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"right, thank you\",\n",
      "        \"intent\": \"affirm\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"correct\",\n",
      "        \"intent\": \"affirm\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"great choice\",\n",
      "        \"intent\": \"affirm\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"sounds really good\",\n",
      "        \"intent\": \"affirm\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"i'm looking for a place to eat\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"I want to grab lunch\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"I am searching for a dinner spot\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"i'm looking for a place in the north of town\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 31,\n",
      "            \"end\": 36,\n",
      "            \"value\": \"north\",\n",
      "            \"entity\": \"location\"\n",
      "          }\n",
      "        ]\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"show me chinese restaurants\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 8,\n",
      "            \"end\": 15,\n",
      "            \"value\": \"chinese\",\n",
      "            \"entity\": \"cuisine\"\n",
      "          }\n",
      "        ]\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"show me chines restaurants in the north\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 8,\n",
      "            \"end\": 14,\n",
      "            \"value\": \"chinese\",\n",
      "            \"entity\": \"cuisine\"\n",
      "          },\n",
      "          {\n",
      "            \"start\": 34,\n",
      "            \"end\": 39,\n",
      "            \"value\": \"north\",\n",
      "            \"entity\": \"location\"\n",
      "          }\n",
      "        ]\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"show me a mexican place in the centre\", \n",
      "        \"intent\": \"restaurant_search\", \n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 31, \n",
      "            \"end\": 37, \n",
      "            \"value\": \"centre\", \n",
      "            \"entity\": \"location\"\n",
      "          }, \n",
      "          {\n",
      "            \"start\": 10, \n",
      "            \"end\": 17, \n",
      "            \"value\": \"mexican\", \n",
      "            \"entity\": \"cuisine\"\n",
      "          }\n",
      "        ]\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"i am looking for an indian spot called olaolaolaolaolaola\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 20,\n",
      "            \"end\": 26,\n",
      "            \"value\": \"indian\",\n",
      "            \"entity\": \"cuisine\"\n",
      "          }\n",
      "        ]\n",
      "      },     {\n",
      "        \"text\": \"search for restaurants\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"anywhere in the west\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 16,\n",
      "            \"end\": 20,\n",
      "            \"value\": \"west\",\n",
      "            \"entity\": \"location\"\n",
      "          }\n",
      "        ]\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"anywhere near 18328\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 14,\n",
      "            \"end\": 19,\n",
      "            \"value\": \"18328\",\n",
      "            \"entity\": \"location\"\n",
      "          }\n",
      "        ]\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"I am looking for asian fusion food\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 17,\n",
      "            \"end\": 29,\n",
      "            \"value\": \"asian fusion\",\n",
      "            \"entity\": \"cuisine\"\n",
      "          }\n",
      "        ]\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"I am looking a restaurant in 29432\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 29,\n",
      "            \"end\": 34,\n",
      "            \"value\": \"29432\",\n",
      "            \"entity\": \"location\"\n",
      "          }\n",
      "        ]\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"I am looking for mexican indian fusion\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 17,\n",
      "            \"end\": 38,\n",
      "            \"value\": \"mexican indian fusion\",\n",
      "            \"entity\": \"cuisine\"\n",
      "          }\n",
      "        ]\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"central indian restaurant\",\n",
      "        \"intent\": \"restaurant_search\",\n",
      "        \"entities\": [\n",
      "          {\n",
      "            \"start\": 0,\n",
      "            \"end\": 7,\n",
      "            \"value\": \"central\",\n",
      "            \"entity\": \"location\"\n",
      "          },\n",
      "          {\n",
      "            \"start\": 8,\n",
      "            \"end\": 14,\n",
      "            \"value\": \"indian\",\n",
      "            \"entity\": \"cuisine\"\n",
      "          }\n",
      "        ]\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"bye\", \n",
      "        \"intent\": \"goodbye\", \n",
      "        \"entities\": []\n",
      "      }, \n",
      "      {\n",
      "        \"text\": \"goodbye\", \n",
      "        \"intent\": \"goodbye\", \n",
      "        \"entities\": []\n",
      "      }, \n",
      "      {\n",
      "        \"text\": \"good bye\", \n",
      "        \"intent\": \"goodbye\", \n",
      "        \"entities\": []\n",
      "      }, \n",
      "      {\n",
      "        \"text\": \"stop\", \n",
      "        \"intent\": \"goodbye\", \n",
      "        \"entities\": []\n",
      "      }, \n",
      "      {\n",
      "        \"text\": \"end\", \n",
      "        \"intent\": \"goodbye\", \n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"farewell\",\n",
      "        \"intent\": \"goodbye\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"Bye bye\",\n",
      "        \"intent\": \"goodbye\",\n",
      "        \"entities\": []\n",
      "      },\n",
      "      {\n",
      "        \"text\": \"have a good one\",\n",
      "        \"intent\": \"goodbye\",\n",
      "        \"entities\": []\n",
      "      }\n",
      "    ]\n",
      "  }\n",
      "}\n"
     ]
    }
   ],
   "source": [
    "# Load Data \n",
    "!cat rasa_dataset.json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading DataSet\n",
    "train_data = load_data('rasa_dataset.json')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Config Backend using Sklearn and Spacy\n",
    "trainer = Trainer(config.load(\"config_spacy.yaml\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Content on Config\n",
    "    language: \"en\"\n",
    "    pipeline: \"spacy_sklearn\"\n",
    "\n",
    "    =======================\n",
    "\n",
    "    language: \"en\"\n",
    "\n",
    "    pipeline:\n",
    "    - name: \"nlp_spacy\"\n",
    "    - name: \"tokenizer_spacy\"\n",
    "    - name: \"intent_entity_featurizer_regex\"\n",
    "    - name: \"intent_featurizer_spacy\"\n",
    "    - name: \"ner_crf\"\n",
    "    - name: \"ner_synonyms\"\n",
    "    - name: \"intent_classifier_sklearn\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 2 folds for each of 6 candidates, totalling 12 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Done  12 out of  12 | elapsed:    0.3s finished\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<rasa_nlu.model.Interpreter at 0x2801960c668>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Training Data\n",
    "trainer.train(train_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Returns the directory the model is stored in (Creat a folder to store model in)\n",
    "model_directory = trainer.persist('/projects/')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Entity Extraction With SpaCy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy\n",
    "nlp = spacy.load('en')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "docx = nlp(u\"I am looking for an Italian Restaurant where I can eat\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "value Italian entity NORP start 20 end 27\n"
     ]
    }
   ],
   "source": [
    "for word in docx.ents:\n",
    "    print(\"value\",word.text,\"entity\",word.label_,\"start\",word.start_char,\"end\",word.end_char)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Making Predictions With Model\n",
    "+ Interpreter.load().parse()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "from rasa_nlu.model import Metadata, Interpreter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# where `model_directory points to the folder the model is persisted in\n",
    "interpreter = Interpreter.load(model_directory)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'intent': {'name': 'restaurant_search', 'confidence': 0.7455215289019911},\n",
       " 'entities': [{'start': 20,\n",
       "   'end': 27,\n",
       "   'value': 'italian',\n",
       "   'entity': 'cuisine',\n",
       "   'confidence': 0.6636828413532201,\n",
       "   'extractor': 'ner_crf'}],\n",
       " 'intent_ranking': [{'name': 'restaurant_search',\n",
       "   'confidence': 0.7455215289019911},\n",
       "  {'name': 'affirm', 'confidence': 0.15019642212447237},\n",
       "  {'name': 'greet', 'confidence': 0.058736824115844515},\n",
       "  {'name': 'goodbye', 'confidence': 0.045545224857692024}],\n",
       " 'text': 'I am looking for an Italian Restaurant where I can eat'}"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Prediction of Intent\n",
    "interpreter.parse(u\"I am looking for an Italian Restaurant where I can eat\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'intent': {'name': 'restaurant_search', 'confidence': 0.6874972430877329},\n",
       " 'entities': [{'start': 10,\n",
       "   'end': 17,\n",
       "   'value': 'african',\n",
       "   'entity': 'cuisine',\n",
       "   'confidence': 0.6470976966769572,\n",
       "   'extractor': 'ner_crf'}],\n",
       " 'intent_ranking': [{'name': 'restaurant_search',\n",
       "   'confidence': 0.6874972430877329},\n",
       "  {'name': 'goodbye', 'confidence': 0.12400667696797882},\n",
       "  {'name': 'affirm', 'confidence': 0.11357435021080386},\n",
       "  {'name': 'greet', 'confidence': 0.07492172973348454}],\n",
       " 'text': 'I want an African Spot to eat'}"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "interpreter.parse(u\"I want an African Spot to eat\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'intent': {'name': 'greet', 'confidence': 0.44328419685532383},\n",
       " 'entities': [],\n",
       " 'intent_ranking': [{'name': 'greet', 'confidence': 0.44328419685532383},\n",
       "  {'name': 'goodbye', 'confidence': 0.31245698090344237},\n",
       "  {'name': 'affirm', 'confidence': 0.1257434275305043},\n",
       "  {'name': 'restaurant_search', 'confidence': 0.11851539471072912}],\n",
       " 'text': 'Good morning World'}"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "interpreter.parse(u\"Good morning World\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Credits Rasa_nlu\n",
    "#### By Jesse JCharis\n",
    "#### Jesus Saves @ JCharisTec"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
