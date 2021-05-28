from emission_probability import find_possible_segments
from transition_probability import calculate_transition_probability
import numpy as np
import time
import networkx as nx
import sys
import os

radius = 50
states = 10

def run_viterbi(g, observations):
    num_of_iterations = len(observations)
    if num_of_iterations == 1:
        segments0, em0 = observations[0][0], observations[0][1]
        em0_vector = np.array(em0)
        max_index = np.argmax(em0)
        return segments0[max_index]['matched-points'], segments0[max_index]['connected-to']

    segments0, em0 = observations[0][0], observations[0][1]
    segments1, em1 = observations[1][0], observations[1][1]

    for i in range(1, num_of_iterations):

        transition_matrix = calculate_transition_probability(g, segments0, segments1)

        if (np.array(transition_matrix)  == -1.7976931348623157e+308).all():
            temp_segments, temp_prob = segments1, em1
            if i == num_of_iterations - 1:
                max_index = np.argmax(em0)   
                temp = (("N/A", ), )
                segments0[max_index]['connected-to'].append(temp)
                temp = (segments1[0]['original-coordinate'],)
                segments0[max_index]['matched-points'].append(temp)
                return segments0[max_index]['matched-points'], segments0[max_index]['connected-to']

            segments1, em1 = observations[i + 1][0], observations[i + 1][1]
            transition_matrix = calculate_transition_probability(g, segments0, segments1)

            if (np.array(transition_matrix)  == -1.7976931348623157e+308).all():
                max_index = np.argmax(em0)
                for j in range(0, len(temp_prob)):
                    temp = temp_segments[j]['connected-to'].pop()
                    temp_segments[j]['connected-to'].extend(segments0[max_index]['connected-to'])
                    temp_segments[j]['connected-to'].append(temp)
                    temp = temp_segments[j]['matched-points'].pop()
                    temp_segments[j]['matched-points'].extend(segments0[max_index]['matched-points'])
                    temp_segments[j]['matched-points'].append(temp)
                segments0 = temp_segments
                em0 = temp_prob
                continue

            else:
                for j in range(0, len(em0)):
                    temp = (("N/A", ), )
                    segments0[j]['connected-to'].append(temp)
                    temp = (temp_segments[0]['original-coordinate'],)
                    segments0[j]['matched-points'].append(temp)
                continue

        for j in range(0, len(em1)):
            transition_vector = np.array(transition_matrix[j])
            if (transition_vector  == -1.7976931348623157e+308).all():
                segments1[j]['emission-probability'] = -1.7976931348623157e+308
                continue
            em0_vector = np.array(em0)
            addition_vector = em0_vector + transition_vector
            max_index = np.argmax(addition_vector)
            em1[j] = em1[j] + addition_vector[max_index]
            segments1[j]['emission-probability'] = segments1[j]['emission-probability'] + addition_vector[max_index]
            temp = segments1[j]['connected-to'].pop()
            segments1[j]['connected-to'].extend(segments0[max_index]['connected-to'])
            segments1[j]['connected-to'].append(temp)
            temp = segments1[j]['matched-points'].pop()
            segments1[j]['matched-points'].extend(segments0[max_index]['matched-points'])
            segments1[j]['matched-points'].append(temp)


        temp_segments = []
        temp_em = []
        count = 0
        for segment in segments1:
            if segment['emission-probability'] != -1.7976931348623157e+308:
                temp_segments.append(segment)
                temp_em.append(em1[count])
            count = count + 1

        em0 = temp_em
        segments0 = temp_segments
        em1 = temp_em
        segments1 = temp_segments

        if i != num_of_iterations - 1:
            segments1, em1 = observations[i + 1][0], observations[i + 1][1]           
            continue
        final_vector = np.array(em1)
        max_index = np.argmax(final_vector)
        return segments1[max_index]['matched-points'], segments1[max_index]['connected-to']




