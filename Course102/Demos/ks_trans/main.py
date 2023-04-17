import copy
import json

from KSParser import KSParser
from LUISKnowledgeStore import LUISKnowledgeStore
import requests
from requests.auth import HTTPDigestAuth

<<<<<<< HEAD

def save(training_path_pattern, test_path_pattern, index, outputter):

    print("Save for index:", index, outputter.outputInformation())

=======

def save(training_path_pattern, test_path_pattern, index, outputter):

    print("Save for index:", index, outputter.outputInformation())

>>>>>>> 0ff8822a8a9f2c4e8c5ccb3ce688fe911ecd5e35
    path = training_path_pattern.replace("{index}", str(index))
    with open(path, "w") as text_file:
        text_file.write(outputter.toJSON())

    jsons = outputter.utteranceToJSON()
    subIndex = 0

    for json in jsons:
        test_path = test_path_pattern.replace("{index}", str(index))
        test_path = test_path.replace("{subIndex}",str(subIndex))
        with open(test_path, "w") as text_file:
            text_file.write(json)
        subIndex += 1

    return


def test(url, test_result_path, outputter):
    success_path = test_result_path.replace("{result}", "success")
    failure_path = test_result_path.replace("{result}", "failure")

    count = 0
    correctCount = 0
    with open(success_path, "w") as success_file:
        with open(failure_path, "w") as failure_file:
            for utterance in outputter.utterances:
                count += 1
                query = utterance.text
                actualIntent = utterance.intent
                testUrl = url.replace("YOURQUERY", query)

                resp = requests.get(testUrl)
                if resp.ok:
                    jData = json.loads(resp.content)
                    predictIntent = jData["prediction"]["topIntent"]
                    result = "[Failed]: "
                    successful = False
                    if predictIntent == actualIntent:
                        correctCount += 1
                        result = "[Succeed]: "
                        successful = True

<<<<<<< HEAD
                    testLog = result + " query: " + query + " actualIntent: " + actualIntent + " predictedIntent: " + predictIntent + "\n"
=======
                    testLog = result + " query: " + query + " actualIntent: " + actualIntent + " predictedIntent: " + predictIntent
>>>>>>> 0ff8822a8a9f2c4e8c5ccb3ce688fe911ecd5e35

                    if successful:
                        success_file.write(testLog)
                    else:
                        failure_file.write(testLog)

<<<<<<< HEAD
=======




>>>>>>> 0ff8822a8a9f2c4e8c5ccb3ce688fe911ecd5e35
    accurancy = correctCount / count * 100

    print("\n")
    print("Totally " + str(count) + " utternaces tested. Correct number: " + str(correctCount) + ". And the accurancy is " + str(accurancy) + "%.")
    return

if __name__ == '__main__':
    input = "C:\\Users\\jull\\Downloads\\Knowledge\\faqTemplate.xlsx"
    output = "C:\\Users\\jull\\Downloads\\Knowledge\\{ksName}_{index}.json"
    test_output = "C:\\Users\\jull\\Downloads\\Knowledge\\{ksName}_test_{index}_{subIndex}.json"
    test_result_path = "C:\\Users\\jull\\Downloads\\Knowledge\\{ksName}_TestResult_{result}.txt"

    url = "https://lutest.cognitiveservices.azure.cn/luis/prediction/v3.0/apps/70242d69-77c8-4e70-9968-70af830e7d48/slots/staging/predict?verbose=true&show-all-intents=true&log=true&subscription-key=3fbdc1303a59492d9cb2851468edbf19&query=\"YOURQUERY\""
    ksName = "xiaotang"

    output = output.replace("{ksName}", ksName)
    test_output = test_output.replace("{ksName}", ksName)
    test_result_path = test_result_path.replace("{ksName}", ksName)

    parser = KSParser()
    knowledges = parser.parse(input)

    if len(knowledges) == 0:
        print("No knowledge")
        exit(0)

    outputters = []

    index = 0
    outputter = LUISKnowledgeStore(ksName + "_" + str(index))
    outputters.append(outputter)
    for k in knowledges:
        if len(k.extendedQuestions) > 0:

            outputter.addFAQ(k.standardQuestion, k.extendedQuestions)

            if outputter.countIntent() == 500:
                index += 1
                newOutputter = copy.copy(outputter)
                newOutputter.cleanFAQ()
                newOutputter.rename(ksName + "_" + str(index))
                outputters.append(newOutputter)
                outputter = newOutputter

    index = 0
    for outputter in outputters:
        save(output, test_output, index, outputter)
        test(url, test_result_path, outputter)
        index += 1

