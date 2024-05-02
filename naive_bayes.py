#Emmanuel Adebayo
#An implementation of naive bayes
#homework 3


#imported the necessary and acceptable libraries
import numpy as np
import math

#read the data into a numpy matrix
#this function takes a given path as argument
#It returns a tuple of the features and the label
def read_data(path):
    
    #load the file into a numpy array
    data = np.loadtxt(path, dtype='str')

    #return the first two column containing the observed data. x1 and x2
    #return the output label which is the last column
    return data[:, :2], data[:,-1]


#Parameters: the feature values and the output label
#returns a tuple that calculate the conditional probability
#of the features given y and n
def estimate_cond_probs(feature_values, labels):
    #Extract unique values from features x1 and x2
    unique_features_values_w1 = np.unique(feature_values[:,0])
    unique_features_values_w2 = np.unique(feature_values[:,1])
    #occurence of y and n labels
    num_of_y_label = np.sum(labels == 'y')
    num_of_n_label = np.sum(labels == 'n')
    #sum of y and n label counts
    total_num_of_label = len(labels)
    #dictionaries to store p_i|y and p_i|n
    prob_dict_y = {}
    prob_dict_n = {}
    #store the conditional probability in two dictionaries. one for label and one for label n.
    for w1 in unique_features_values_w1:
        prob_dict_y[w1] =  ( count_conditional_prob(feature_values,labels,0,w1,"y") / num_of_y_label ) 
        prob_dict_n[w1] =  (count_conditional_prob(feature_values,labels,0,w1,"n") / num_of_n_label ) 
    for w2 in unique_features_values_w2:
        prob_dict_y[w2] =  (count_conditional_prob(feature_values,labels,1,w2,"y") / num_of_y_label)
        prob_dict_n[w2] =  (count_conditional_prob(feature_values,labels,1,w2,"n") / num_of_n_label)    
    #return the dictionaries as a tuple
    return (prob_dict_y, prob_dict_n)

#an helper function used in estimating cconditional probability
#please refer to the above function for context
def count_conditional_prob(feature_values, labels, column_index, feature_value_to_search, label_to_search):
    count = 0
    for row, value in zip(feature_values, labels):
        #print(row[column_index], value)
        if ( (row[column_index] == feature_value_to_search) and (value == label_to_search)):
            count = count +1
    # print(count)
    return count


#estimates the probability of the output label
def estimate__Y(labels):
     #occurence of y and n labels
    num_of_y_label = np.sum(labels == 'y')
    num_of_n_label = np.sum(labels == 'n')
    #sum of y and n label counts
    total_num_of_label = len(labels)

    prior_y = num_of_y_label / total_num_of_label
    prior_n = num_of_n_label / total_num_of_label

    label_prob_dict = {}
    label_prob_dict['y'] = prior_y
    label_prob_dict['n'] = prior_n
    return label_prob_dict


#parameters: trained model (a tuple of the conditional probability and the probability of the output label)
#parameters: X ( an unseen and unclassified example to be classified) an array
def classify(trained_model, X):
    
    prob_cond_y, prob_cond_n, prior_prob_y = trained_model

    feature_w1 = X[0]
    feature_w2 = X[1]
    try:
        check_if_yes = np.log10(prob_cond_y[feature_w1]) + np.log10(prob_cond_y[feature_w2]) + np.log10(prior_prob_y['y'])

        check_if_no = np.log10(prob_cond_n[feature_w1]) + np.log10(prob_cond_n[feature_w2]) + np.log10(prior_prob_y['n'])
    except:
        print(f"An attempt to take the log zero has caused an error: ")
    if (check_if_yes > check_if_no):
        return "Y"

    return "N"

#returns the log value of the conditional probability
#if it is zero, it returns a string indicating that it can happen
def get_conditional_log_probability(p_i_y, p_i_n, attribute):
    given_y = p_i_y[attribute]
    given_n = p_i_n[attribute]

    if given_y != 0:
        given_y = "the log conditional probability given y of the attribute is " +attribute+ ": " + str (np.log10(given_y) )
    else:
        given_y = "Cannot take the log probability of zero"
    if given_n != 0:
        given_n = "the log conditional probability given n of the attribute is " +attribute+  ": " + str (np.log10(given_n))
    else:
        given_n = "Cannot take the log probability when it is zero"
    

    return given_y, given_n

#returns the log value of label probability
def get_log_prior_probability(y):

    return np.log10(y)


#answers part a
path_to_file = 'training.txt'
features, annotations = read_data(path_to_file)

#answers part b
p_y, p_n = estimate_cond_probs(features,annotations)

#answers part c
attribute = "blue"
if_y, if_n = get_conditional_log_probability(p_y, p_n, attribute)
print(if_y)
print(if_n)

#answers part d
attribute = "cat"
if_y, if_n = get_conditional_log_probability(p_y, p_n, attribute)
print(if_y)
print(if_n)

#answers part e
estimate_text = 'y'
estimate_text_n = 'n'
labels = estimate__Y(annotations)
print("the prior probability for class "+ estimate_text+ " is "+str(labels[estimate_text]))
print("the prior probability for class "+ estimate_text_n+ " is "+str(labels[estimate_text_n]))

#answrs part f
print("the log prior probability for class "+ estimate_text+ " is "+ str(get_log_prior_probability(labels[estimate_text])))
print("the log prior probability for class "+ estimate_text_n+ " is "+ str(get_log_prior_probability(labels[estimate_text_n])))

#answrs part g and h
#model the unlabeled_new_feature to test new features
tuple_of_trained_model = p_y, p_n, estimate__Y(annotations)
unlabeled_new_feature = ["cautious", "gloves"]
print ( classify(tuple_of_trained_model, unlabeled_new_feature) )