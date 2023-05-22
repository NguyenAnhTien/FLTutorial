"""
@author : Tien Nguyen
@date   : 2023-05-22
"""
import constants

def scale_params(
        params,
        fraction, 
    ) -> dict:
    scaled_params = {}
    for layer in params:
        scale_params[layer] = {
            constants.BIAS   : 0,
            constants.WEIGHT : 0
        }
    for layer in params:
        scaled_params[layer][constants.WEIGHT] = \
                                    fraction * params[layer][constants.WEIGHT]
        scaled_params[layer][constants.BIAS] = \
                                    fraction * params[layer][constants.BIAS]
    return scaled_params

def sum_weights(
        params_list: list
    ) -> None:
    sum_params = {}
    for layer in params_list[0]:
        sum_params[layer] = {
            constants.BIAS   : 0,
            constants.WEIGHT : 0
        }
    for index in range(len(params_list)):
        for layer in params_list[index]:
            weights = params_list[index][layer][constants.WEIGHT]
            bias = params_list[index][layer][constants.BIAS]
            sum_params[layer][constants.WEIGHT] += weights
            sum_params[layer][constants.BIAS] += bias
    
    return sum_params
