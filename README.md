# AIR Data Scientist Associate Assignment

This assignment represents some of the machine learning and data science skills I have learned over my first five years of coding. This assessment encompasses a broad range of problems including building and evaluating machine learning models, algorithms, and data cleaning.

1. Machine Learning
Imagine that we are trying to create a system to help us identify a rare genetic mutation. Our system will use machine learning on various measurements that have been collected to predict whether a person has this mutation or not. This “gene” data set includes 28 anonymized variables and a “Class” variable that is 1 if the mutation is present and 0 if it is not. Use this data to answer the following questions.
In this problem, I used a Logistic Regression and Random Forest Classifier to predict gene mutations. After evaluating both models, we found our Recall scores were particularly low, meaning our model was not accurately predicting gene mutations. I believe oversampling of negative instances was interfering with our model making accurate predictions. After undersampling our negative instances, our Random Forest Classifier fared far better at predicting gene mutations.

2. Implementing an Edit-Distance Algorithm
Write a program to calculate a variant of the Hamming distance with two key modifications to the standard algorithm. In information theory, the Hamming distance is a measure of the distance between two text strings. This is calculated by adding one to the Hamming distance for each character that is different between the two strings. For example, “kitten" and “mitten" have a Hamming distance of 1. See https://en.wikipedia.org/wiki/Hamming_distance for more information. 
Modifications to the standard Hamming distance algorithm for the purposes of this exercise include: 
- Add .5 to the Hamming distance if a capital letter is switched for a lower case letter unless it is in the first position.  Examples include: 
- "Kitten" and "kitten" have a distance of 0 
- "kitten" and "KiTten" have a Hamming distance of .5.
- "Puppy" and "POppy" have a distance of 1.5 (1 for the different letter, additional .5 for the different capitalization). 
- Consider S and Z (and s and z) to be the same letter. For example, "analyze" has a distance of 0 from "analyse".
You might expect Hamming distance algorithms to be used in expected use cases like auto-corrections or spell-checking. We might also expect Hamming distance where sequencing is important.

3. Data Cleaning
Perform some data cleaning using the provided file, “patent_drawing.csv”. “Patent_drawing.csv” contains a list of patents and a short description of each drawing included with a patent grant. For example, patent number 0233365 has 16 images. For each image, there is a brief description of the drawings. The description is included in the “text” field in patent_drawing.csv. 
In this problem, we searched our file for keywords and phrases to calculate and return all of the patents that meet our criteria. We also used Regular Expressions to identify more complex phrases.
