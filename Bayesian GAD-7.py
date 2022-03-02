import numpy as np
import sys
array = np.genfromtxt('/Users/bennettliu/Desktop/CS109/Final Project/Data GAD-7.csv', delimiter=",", dtype = int)
from matplotlib import pyplot as plt
#Data filtering and processing

#I first delete the first row because this corresponds to the header
array = np.delete(array,(0),axis=0)
np.set_printoptions(threshold=sys.maxsize)


#Next, I look into the dataset and see that some of the data has rows of 8. Since 8 is not a possible
#score on the GAD, then I remove all rows with values that are greater than 8
filteredarray = []
for i in range(len(array)):
    keep = True
    for j in range(len(array[i])):
       if array[i][j] >= 8:
           keep = False
    if keep == True:
        filteredarray.append(array[i])
responses = np.asarray(filteredarray)


#Now we can create the dictionaries that store the values with the probabilities. First, I am going
#to create a dictionary of all of the marginalization factors. These are for example, P(Q1 = 1), the
#probability that someone answers Q1 with a 1. The first indicy of the key corresponds to the question number (Q1)
#the second indicy corresponds to the answer number (Answered 1 on Q1)
marginalizations = {}

for i in range(1,8): #question number
    for j in range(1,5): #subject answered value
        total = 0
        for person in responses:
            if person[i] == j:
                total += 1
        """ if total == 0:#if a total has 0, we add 1 so that we avoid division by 0 when we use this in bayes
            total+=1 """
        marginalizations[str(i)+str(j)] = total/len(responses)


#Next, we can create the dictionary that holds the priors. This dictionary will be 4 entries, and will
#hold the prior belief in the population that someone has anxiety. For example, P(A=1) corresponds to 
#the probability that someone has an anxiety level corresponding to value of 1 (mild or no anxiety). The
# indicy of the key will correspond to the anxiety level
priors = {}
priortotals = {}

for i in range(1,5):# all of the answer possibilities
    total = 0
    for person in responses: #every entry in our responses
        if person[7] == i:
            total+=1
    priors[str(i)] = total/len(responses)
    priortotals[str(i)] = total
#Finally, this dictionary will correspond to the likelihoods. For example this dictionary will
#hold P(Q1 = 2| A = 1). What is the probability that someone answers 2 for question 1 given
#they have an anxiety level of 1? To calculate this probability, I will use the definiton of
#conditional probability. P(E|F) = P(EF)\P(F). So, I will first calculate the probability
#that someone answers 2 for question 1 and they have an anxiety level of 1, then divide this
#value by the probability that their anxiety is a level 1. The first indicy of the dictionary will
#correspond to the question number (1-7), the second indicy will corrrespond to the answer to
#the question (1-4), and the third indicy corresponds to the anxiety level (1-4)
#For example, likelihood[121] would be Q1, the participant answered 2, and their prior anxiety level is 2
likelihoods = {}
for i in range(1,8):#corresponding to question number
    for j in range(1,5):#corrresponding to the answer of the question
        for k in range(1,5):#corresponding to the prior anxiety level of the participant
            total = 0
            for person in responses: #every entry in our responses
                if person[i-1] == j and person[7] == k:
                    total+=1
            likelihoods[str(i)+str(j)+str(k)] = total/(priortotals[str(k)])


#This function calculates the posterior based on the question number, question answer, and the anxiety level
#Uses Bayes Thereom to accomplish this
def posteriorNum(qNum, qAnswer, anxietyLevel, indPriors):
    """ print("likelihood")
    print(likelihoods[str(qNum) + str(qAnswer) + str(anxietyLevel)])
    print("prior")
    print(anxietyLevel)
    print("marginalization")
    print(marginalizations[str(qNum) + str(qAnswer)]) """
    return (likelihoods[str(qNum) + str(qAnswer) + str(anxietyLevel)]*indPriors[str(anxietyLevel)])

#this function takes in a posterior and returns the normalized version of it.
def normalize(posterior):
    total = 0
    for i in range(1,len(posterior)+1):
        total+= posterior[str(i)]
    normConst = 1/total
    normalizePost = {}
    for i in range(1,len(posterior)+1):
        normalizePost[str(i)] = posterior[str(i)]*normConst
    return normalizePost


def main():
    #plot the prior belief in individual's anxiety
    plt.subplot(2,4,1)
    xCords = [1,2,3,4]
    yCords = [priors[str(1)],priors[str(2)],priors[str(3)],priors[str(4)]]
    plt.grid()
    plt.xlabel("Anxiety Severity")
    plt.ylabel("Probability of Anxiety Severity")
    plt.scatter(xCords,yCords)
    plt.title('Population Average Score')
    indPriors = priors.copy()
    
    print("\n \n \n Thank you for taking the GAD-7. The GAD-7 is a diagnostic test used for screening anxiety. \n We have redesigned the test to give a distribution of your likelihood of having anxiety. \n You will be asked 7 questions. In each question, please type in a number from 1-4 corresponding \n to the number of days of the past two weeks you have been experiencing the anxiety symptom. \n \n 1: Not at all \n 2: Several days \n 3 More than half the days \n 4: Nearly Everyday. \n \n Please try to answer the questions to the best of your ability, but no worries if you feel conflicted between two answers. \nAt the end of the quiz, we will give you a graph that corresponds to your likelihood of having anxiety.\n And, if you are looking for more mental health resources, please visit: https://vaden.stanford.edu/caps \n")
    answer1 = input("Feeling nervous, anxious, or on edge:\n")
    answer2 = input("Not being able to stop or control worrying:\n")
    answer3 = input("Worrying too much about different things:\n")
    answer4 = input("Trouble relaxing:\n")
    answer5 = input("Being so restless that it is hard to sit still:\n")
    answer6 = input("Becoming easily annoyed or irritable:\n")
    answer7 =input("Feeling afraid, as if something aweful might happen:\n")
    answers =[answer1,answer2,answer3,answer4,answer5,answer6,answer7]

    #for loop that will take the users entries and update the beliefs in the anxiety distribution
    for i in range(1,8):#goes through each of the questions
         for j in range(1,5):#goes through each of the prior anxiety levels and recomputes
            indPriors[str(j)] = posteriorNum(i,answers[i-1],j, indPriors)
         normIndPriors = normalize(indPriors)
         indPriors = normIndPriors.copy()
         plt.subplot(2,4,i+1)
         xCords = [1,2,3,4]
         yCords = [normIndPriors[str(1)],normIndPriors[str(2)],normIndPriors[str(3)],normIndPriors[str(4)]]
         plt.grid()
         plt.xlabel("Anxiety Severity")
         plt.ylabel("Probability of Anxiety Severity")
         plt.scatter(xCords,yCords)
         if i == 7:
             plt.title('Final Score')
             break
         plt.title('Score After Question '+ str(i))
    plt.tight_layout()
    plt.show()
    print('Thank you for taking the GAD-7. The anxiety severity scores correspond as follows: \n\n 1: Minimal Anxiety \n 2: Mild Anxiety \n 3: Moderate Anxiety \n 4:Severe Anxiety \n \n If you would like to follow up on your results for an official diagnosis, \n please reach out to CAPS: https://vaden.stanford.edu/caps')

if __name__ == "__main__":
    main()


