import copy
import json

class Utterance:
    def __init__(self, text, intent, entities=[]):
        self.text =text
        self.intent = intent
        self.entities = entities
        return


class Intent:
    def __init__(self, name, features=[]):
        self.name = name
        self.features = features

class LUISKnowledgeStore:
    def __init__(self, name, desc="", culture="zh-cn"):
        self.luis_schema_version = "7.0.0"
        self.versionId = "0.1"
        self.name = name
        self.desc = desc
        self.culture = culture
        self.tokenizerVersion = "1.0.0"
        self.intents = []
        self.utterances = []

        self.entities = []
        self.hierarchicals = []
        self.composites = []
        self.closedLists = []
        self.prebuiltEntities = []
        self.patternAnyEntities = []
        self.regex_entities = []
        self.phraselists = []
        self.regex_features = []
        self.patterns = []
        self.settings = []

        return

    def rename(self, newName):
        self.name = newName
        return

    def addFAQ(self, standardQuestion, extendesQuestions):
        intentName = standardQuestion
        intent = Intent(intentName)
        self.intents.append(intent)
        if len(extendesQuestions) > 0:
            for eq in extendesQuestions:
                utternace = Utterance(eq, intentName)
                self.utterances.append(utternace)
        return

    def cleanFAQ(self):
        self.intents.clear()

    def countIntent(self):
        return len(self.intents)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4, ensure_ascii=False)

    def utteranceToJSON(self):
        jsonList = []
        count = 1000
        index = 0
        while index < len(self.utterances):
            start = index
            end = index + count

            if end > len(self.utterances):
                end = -1

            aJson = json.dumps(self.utterances[start:end], default=lambda o: o.__dict__, sort_keys=True, indent=4, ensure_ascii=False)
            jsonList.append(aJson)
            index += count

        return jsonList

    def outputInformation(self):
        intentNum = len(self.intents)
        utteranceNum = len(self.utterances)

        return "KS: " + self.name + " with " + str(intentNum) + " intents and " + str(utteranceNum) + " utterances."
