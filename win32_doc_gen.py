
from datetime import datetime
from query import run_dxc, query_dxil, get_intrinsic_param_types, load_dict
from utils import ApiKeys
from openai import OpenAI
import json
import os
import pathlib

undocumented_apis = [
    "NonUniformResourceIndex",
    "AddUint64",
    "GetAttributeAtVertex",
    "asfloat16",
    "asint16",
    "asuint16",
    "dot4add_u8packed",
    "dot4add_i8packed",
    "dot2add",
    "WaveMultiPrefixCountBits",
    "WaveMultiPrefixProduct",
    "WaveMatch",
    "WaveMultiPrefixBitAnd",
    "WaveMultiPrefixBitOr",
    "WaveMultiPrefixBitXor",
    "WaveMultiPrefixSum",
    "IsHelperLane",
    "QuadAny",
    "QuadAll",
    "unpack_s8s16",
    "unpack_u8u16",
    "unpack_s8s32",
    "unpack_u8u32",
    "pack_s8",
    "pack_u8",
    "pack_clamp_s8",
    "pack_clamp_u8",
    "SetMeshOutputCounts",
    "DispatchMesh",
    "AllocateRayQuery",
    "CreateResourceFromHeap",
    #"and",
    #"or",
    "select",
    "Barrier",
    "GetRemainingRecursionLevels",
]

component_type_dict = {'float_like<>' : 'float or double',
                       'float16_t<>'  : 'half',
                       'numeric<>' : 'float or int',
                       'numeric16_only<>' : 'half, int16_t, or uint16_t',
                       'float' : 'float',
                       'float16_t<2>' : 'half',
                       'int' : 'int',
                       'int<4>' : 'int',
                       'uint' : 'uint',
                       'uint<4>' : 'uint',
                       'uint<c>' : 'uint',
                       'int16_t<4>' : 'int16_t',
                       'int16_t<>' : 'int16_t',
                       'uint16_t<>' : 'uint16_t',
                       'uint16_t<4>' : 'uint16_t',
                       'any_int16or32<4>' : 'int, int16_t, uint, or uint16_t',
                       'sint16or32_only<4>' : 'int or int16_t',
                       'any<>' : 'bool, float, or int',
                       'any_int<>' : 'int or uint',
                       'any_sampler' : 'float, int, or uint',
                       'bool' : 'bool',
                       'p32i8' : 'int8_t4_packed',
                       'p32u8' : 'uint8_t4_packed',
                       'void' : 'void',
                       'udt' : 'RayPayload',
                       'resource' : 'resource'
                      }

template_type_dict = {'float_like<>' : 'scalar, vector, or matrix',
                   'float16_t<>' :  'scalar, vector, or matrix',
                   'numeric<>'  :   'scalar, vector, or matrix',
                   'numeric16_only<>' : 'scalar, vector, or matrix',
                   'float' : 'scalar',
                   'float16_t<2>' : 'vector size 2',
                   'int'  : 'scalar',
                   'int<4>' : 'vector',
                   'uint'  : 'scalar',
                   'uint<4>' : 'vector size 4',
                   'uint<c>' : 'vector',
                   'int16_t<4>' : 'vector size 4',
                   'int16_t<>' : 'scalar, vector, or matrix',
                   'uint16_t<>' : 'scalar, vector, or matrix',
                   'uint16_t<4>' : 'vector size 4',
                   'any_int16or32<4>' : 'vector size 4',
                   'sint16or32_only<4>' : 'vector size 4',
                   'any<>' : 'scalar, vector, or matrix',
                   'any_int<>' : 'scalar, vector, or matrix',
                   'any_sampler' : 'scalar, vector, or matrix',
                   'bool' : 'scalar',
                   'p32i8' : '4 byte packed scalar',
                   'p32u8' : '4 byte packed scalar',
                   'void' : 'void',
                   'udt' : "RayPayload Struct",
                   'resource' : 'resource'
                  }

def load_cache(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {}

def save_cache(file_path, cache):
    with open(file_path, 'w') as file:
        json.dump(cache, file, indent=4)

def ask_openai_to_document_hlsl_intrinsic(intrinsic_name, params):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an assistant that documents HLSL intrinsics."},
        {"role": "user", 
            "content": 
            f"Document the HLSL intrinsic function '{intrinsic_name}' which takes {len(params)-1 }" + 
            f"parameters and a return template type of {template_type_dict[params['ret']]} and component type of {component_type_dict[params['ret']]}." 
            + "Provide descriptions for each parameter and the return type." +
            f" Use the parameter names {', '.join([param_name for param_name in params.keys() if param_name != 'ret'])}."
            }
    ],
    functions=[
        {
            "name": "document_intrinsic",
            "description": "Document an HLSL intrinsic function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intrinsic_name": {
                        "type": "string",
                        "description": "The name of the HLSL intrinsic function."
                    },
                    "parameter_descriptions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "The name of the parameter."},
                                "description": {"type": "string", "description": "A description of the parameter."}
                            }
                        }
                    },
                    "return_description": {
                        "type": "string",
                        "description": "A description of the return type."
                    }
                },
                "required": ["intrinsic_name"]
            }
        }
    ])

   # Extract the function call
    function_call = response.choices[0].message.function_call
    # Parse the arguments as JSON
    arguments = json.loads(function_call.arguments)

    return arguments

def document_hlsl_intrinsic(intrinsic_name, params):
    scratchpad_path = os.path.join(pathlib.Path().resolve(), 'scratch')
    os.makedirs(scratchpad_path, exist_ok=True)
    file_path = os.path.join(scratchpad_path, 'open_ai_doc.json')
    cache = load_cache(file_path)

    if intrinsic_name in cache:
        return cache[intrinsic_name]

    # Key does not exist, call the expensive function
    data = ask_openai_to_document_hlsl_intrinsic(intrinsic_name, params)
    if data:
        cache[intrinsic_name] = data
        save_cache(file_path, cache)
    return data

shader_functions = {
    "abort": {"Description": "Terminates the current draw or dispatch call being executed.", "Minimum shader model": 4},
    "abs": {"Description": "Absolute value (per component).", "Minimum shader model": 1},
    "acos": {"Description": "Returns the arccosine of each component of x.", "Minimum shader model": 1},
    "all": {"Description": "Test if all components of x are nonzero.", "Minimum shader model": 1},
    "AllMemoryBarrier": {"Description": "Blocks execution of all threads in a group until all memory accesses have been completed.", "Minimum shader model": 5},
    "AllMemoryBarrierWithGroupSync": {"Description": "Blocks execution of all threads in a group until all memory accesses have been completed and all threads in the group have reached this call.", "Minimum shader model": 5},
    "any": {"Description": "Test if any component of x is nonzero.", "Minimum shader model": 1},
    "asdouble": {"Description": "Reinterprets a cast value into a double.", "Minimum shader model": 5},
    "asfloat": {"Description": "Convert the input type to a float.", "Minimum shader model": 4},
    "asin": {"Description": "Returns the arcsine of each component of x.", "Minimum shader model": 1},
    "asint": {"Description": "Convert the input type to an integer.", "Minimum shader model": 4},
    "asuint": {"Description": "Reinterprets the bit pattern of a 64-bit type to a uint.", "Minimum shader model": 5},
    "asuint": {"Description": "Convert the input type to an unsigned integer.", "Minimum shader model": 4},
    "atan": {"Description": "Returns the arctangent of x.", "Minimum shader model": 1},
    "atan2": {"Description": "Returns the arctangent of two values (x,y).", "Minimum shader model": 1},
    "ceil": {"Description": "Returns the smallest integer which is greater than or equal to x.", "Minimum shader model": 1},
    "CheckAccessFullyMapped": {"Description": "Determines whether all values from a Sample or Load operation accessed mapped tiles in a tiled resource.", "Minimum shader model": 5},
    "clamp": {"Description": "Clamps x to the range [min, max].", "Minimum shader model": 1},
    "clip": {"Description": "Discards the current pixel, if any component of x is less than zero.", "Minimum shader model": 1},
    "cos": {"Description": "Returns the cosine of x.", "Minimum shader model": 1},
    "cosh": {"Description": "Returns the hyperbolic cosine of x.", "Minimum shader model": 1},
    "countbits": {"Description": "Counts the number of bits (per component) in the input integer.", "Minimum shader model": 5},
    "cross": {"Description": "Returns the cross product of two 3D vectors.", "Minimum shader model": 1},
    "D3DCOLORtoUBYTE4": {"Description": "Swizzles and scales components of the 4D vector xto compensate for the lack of UBYTE4 support in some hardware.", "Minimum shader model": 1},
    "ddx": {"Description": "Returns the partial derivative of x with respect to the screen-space x-coordinate.", "Minimum shader model": 2},
    "ddx_coarse": {"Description": "Computes a low precision partial derivative with respect to the screen-space x-coordinate.", "Minimum shader model": 5},
    "ddx_fine": {"Description": "Computes a high precision partial derivative with respect to the screen-space x-coordinate.", "Minimum shader model": 5},
    "ddy": {"Description": "Returns the partial derivative of x with respect to the screen-space y-coordinate.", "Minimum shader model": 2},
    "ddy_coarse": {"Description": "Computes a low precision partial derivative with respect to the screen-space y-coordinate.", "Minimum shader model": 5},
    "ddy_fine": {"Description": "Computes a high precision partial derivative with respect to the screen-space y-coordinate.", "Minimum shader model": 5},
    "degrees": {"Description": "Converts x from radians to degrees.", "Minimum shader model": 1},
    "determinant": {"Description": "Returns the determinant of the square matrix m.", "Minimum shader model": 1},
    "DeviceMemoryBarrier": {"Description": "Blocks execution of all threads in a group until all device memory accesses have been completed.", "Minimum shader model": 5},
    "DeviceMemoryBarrierWithGroupSync": {"Description": "Blocks execution of all threads in a group until all device memory accesses have been completed and all threads in the group have reached this call.", "Minimum shader model": 5},
    "distance": {"Description": "Returns the distance between two points.", "Minimum shader model": 1},
    "dot": {"Description": "Returns the dot product of two vectors.", "Minimum shader model": 1},
    "dst": {"Description": "Calculates a distance vector.", "Minimum shader model": 5},
    "errorf": {"Description": "Submits an error message to the information queue.", "Minimum shader model": 4},
    "EvaluateAttributeCentroid": {"Description": "Evaluates at the pixel centroid.", "Minimum shader model": 5},
    "EvaluateAttributeAtSample": {"Description": "Evaluates at the indexed sample location.", "Minimum shader model": 5},
    "EvaluateAttributeSnapped": {"Description": "Evaluates at the pixel centroid with an offset.", "Minimum shader model": 5},
    "exp": {"Description": "Returns the base-e exponent.", "Minimum shader model": 1},
    "exp2": {"Description": "Base 2 exponent (per component).", "Minimum shader model": 1},
    "f16tof32": {"Description": "Converts the float16 stored in the low-half of the uint to a float.", "Minimum shader model": 5},
    "f32tof16": {"Description": "Converts an input into a float16 type.", "Minimum shader model": 5},
    "faceforward": {"Description": "Returns -n * sign(dot(i, ng)).", "Minimum shader model": 1},
    "firstbithigh": {"Description": "Gets the location of the first set bit starting from the highest order bit and working downward, per component.", "Minimum shader model": 5},
    "firstbitlow": {"Description": "Returns the location of the first set bit starting from the lowest order bit and working upward, per component.", "Minimum shader model": 5},
    "floor": {"Description": "Returns the greatest integer which is less than or equal to x.", "Minimum shader model": 1},
    "fma": {"Description": "Returns the double-precision fused multiply-addition of a * b + c.", "Minimum shader model": 5},
    "fmod": {"Description": "Returns the floating point remainder of x/y.", "Minimum shader model": 1},
    "frac": {"Description": "Returns the fractional part of x.", "Minimum shader model": 1},
    "frexp": {"Description": "Returns the mantissa and exponent of x.", "Minimum shader model": 2},
    "fwidth": {"Description": "Returns abs(ddx(x)) + abs(ddy(x))", "Minimum shader model": 2},
    "GetRenderTargetSampleCount": {"Description": "Returns the number of render-target samples.", "Minimum shader model": 4},
    "GetRenderTargetSamplePosition": {"Description": "Returns a sample position (x,y) for a given sample index.", "Minimum shader model": 4},
    "GroupMemoryBarrier": {"Description": "Blocks execution of all threads in a group until all group shared accesses have been completed.", "Minimum shader model": 5},
    "GroupMemoryBarrierWithGroupSync": {"Description": "Blocks execution of all threads in a group until all group shared accesses have been completed and all threads in the group have reached this call.", "Minimum shader model": 5},
    "InterlockedAdd": {"Description": "Performs a guaranteed atomic add of value to the dest resource variable.", "Minimum shader model": 5},
    "InterlockedAnd": {"Description": "Performs a guaranteed atomic and.", "Minimum shader model": 5},
    "InterlockedCompareExchange": {"Description": "Atomically compares the input to the comparison value and exchanges the result.", "Minimum shader model": 5},
    "InterlockedCompareStore": {"Description": "Atomically compares the input to the comparison value.", "Minimum shader model": 5},
    "InterlockedExchange": {"Description": "Assigns value to dest and returns the original value.", "Minimum shader model": 5},
    "InterlockedMax": {"Description": "Performs a guaranteed atomic max.", "Minimum shader model": 5},
    "InterlockedMin": {"Description": "Performs a guaranteed atomic min.", "Minimum shader model": 5},
    "InterlockedOr": {"Description": "Performs a guaranteed atomic or.", "Minimum shader model": 5},
    "InterlockedXor": {"Description": "Performs a guaranteed atomic xor.", "Minimum shader model": 5},
    "isfinite": {"Description": "Returns true if x is finite, false otherwise.", "Minimum shader model": 1},
    "isinf": {"Description": "Returns true if x is +INF or -INF, false otherwise.", "Minimum shader model": 1},
    "isnan": {"Description": "Returns true if x is NAN or QNAN, false otherwise.", "Minimum shader model": 1},
    "ldexp": {"Description": "Returns x * 2exp.", "Minimum shader model": 2},
    "length": {"Description": "Returns the length of the vector.", "Minimum shader model": 1},
    "lerp": {"Description": "Returns the linear interpolation of x and y based on the value of s.", "Minimum shader model": 1},
    "lit": {"Description": "Returns a lighting vector (ambient, diffuse, specular, 1) based on the dot products of the input vectors.", "Minimum shader model": 1},
    "log": {"Description": "Returns the base-e logarithm of x.", "Minimum shader model": 1},
    "log10": {"Description": "Returns the base-10 logarithm of x.", "Minimum shader model": 1},
    "log2": {"Description": "Returns the base-2 logarithm of x.", "Minimum shader model": 1},
    "mad": {"Description": "Returns (a * b) + c.", "Minimum shader model": 2},
    "max": {"Description": "Selects the greater of x or y.", "Minimum shader model": 1},
    "min": {"Description": "Selects the lesser of x or y.", "Minimum shader model": 1},
    "modf": {"Description": "Splits the value x into fractional and integer parts.", "Minimum shader model": 2},
    "msad4": {"Description": "Computes a sum of absolute differences (SAD) with 4-bit packed unsigned values.", "Minimum shader model": 5},
    "mul": {"Description": "Returns the result of a matrix multiplication.", "Minimum shader model": 1},
    "normalize": {"Description": "Returns a normalized vector.", "Minimum shader model": 1},
    "pow": {"Description": "Returns x^n.", "Minimum shader model": 1},
    "printf": {"Description": "Submits a custom shader message to the information queue.", "Minimum shader model": 4},
    "Process2DQuadTessFactorsAvg": {"Description": "Processes the quad tessellation factors and outputs valid factors.", "Minimum shader model": 5},
    "Process2DQuadTessFactorsMax": {"Description": "Processes the quad tessellation factors and outputs valid factors.", "Minimum shader model": 5},
    "Process2DQuadTessFactorsMin": {"Description": "Processes the quad tessellation factors and outputs valid factors.", "Minimum shader model": 5},
    "ProcessIsolineTessFactors": {"Description": "Processes the isoline tessellation factors and outputs valid factors.", "Minimum shader model": 5},
    "ProcessQuadTessFactorsAvg": {"Description": "Processes the quad tessellation factors and outputs valid factors.", "Minimum shader model": 5},
    "ProcessQuadTessFactorsMax": {"Description": "Processes the quad tessellation factors and outputs valid factors.", "Minimum shader model": 5},
    "ProcessQuadTessFactorsMin": {"Description": "Processes the quad tessellation factors and outputs valid factors.", "Minimum shader model": 5},
    "ProcessTriTessFactorsAvg": {"Description": "Processes the quad tessellation factors and outputs valid factors.", "Minimum shader model": 5},
    "ProcessTriTessFactorsMax": {"Description": "Processes the triangle tessellation factors and outputs valid factors.", "Minimum shader model": 5},
    "ProcessTriTessFactorsMin": {"Description": "Processes the triangle tessellation factors and outputs valid factors.", "Minimum shader model": 5},
    "radians": {"Description": "Converts x from degrees to radians.", "Minimum shader model": 1},
    "rcp": {"Description": "Per-component reciprocal.", "Minimum shader model": 1},
    "reflect": {"Description": "Returns a reflection vector.", "Minimum shader model": 1},
    "refract": {"Description": "Returns a refraction vector.", "Minimum shader model": 1},
    "reversebits": {"Description": "Reverses the order of the bits.", "Minimum shader model": 5},
    "round": {"Description": "Rounds x to the nearest integer.", "Minimum shader model": 1},
    "rsqrt": {"Description": "Reciprocal square root (per component).", "Minimum shader model": 1},
    "saturate": {"Description": "Clamps x to the range [0, 1].", "Minimum shader model": 1},
    "sign": {"Description": "Returns the sign of x.", "Minimum shader model": 1},
    "sin": {"Description": "Returns the sine of x.", "Minimum shader model": 1},
    "sincos": {"Description": "Returns the sine and cosine of x.", "Minimum shader model": 1},
    "sinh": {"Description": "Returns the hyperbolic sine of x.", "Minimum shader model": 1},
    "smoothstep": {"Description": "Returns a smooth Hermite interpolation.", "Minimum shader model": 1},
    "sqrt": {"Description": "Square root (per component).", "Minimum shader model": 1},
    "step": {"Description": "Compares two values.", "Minimum shader model": 1},
    "tan": {"Description": "Returns the tangent of x.", "Minimum shader model": 1},
    "tanh": {"Description": "Returns the hyperbolic tangent of x.", "Minimum shader model": 1},
    "tex1D": {"Description": "Samples a 1D texture.", "Minimum shader model": 1},
    "tex1Dbias": {"Description": "Samples a 1D texture after biasing the mip level by the specified value.", "Minimum shader model": 1},
    "tex1Dgrad": {"Description": "Samples a 1D texture using a gradient to select the mip level.", "Minimum shader model": 1},
    "tex1Dlod": {"Description": "Samples a 1D texture at a specified mip level.", "Minimum shader model": 1},
    "tex1Dproj": {"Description": "Samples a 1D texture and projects the texture coordinate by dividing it by the last component.", "Minimum shader model": 1},
    "tex2D": {"Description": "Samples a 2D texture.", "Minimum shader model": 1},
    "tex2Dbias": {"Description": "Samples a 2D texture after biasing the mip level by the specified value.", "Minimum shader model": 1},
    "tex2Dgrad": {"Description": "Samples a 2D texture using a gradient to select the mip level.", "Minimum shader model": 1},
    "tex2Dlod": {"Description": "Samples a 2D texture at a specified mip level.", "Minimum shader model": 1},
    "tex2Dproj": {"Description": "Samples a 2D texture and projects the texture coordinate by dividing it by the last component.", "Minimum shader model": 1},
    "tex3D": {"Description": "Samples a 3D texture.", "Minimum shader model": 1},
    "tex3Dbias": {"Description": "Samples a 3D texture after biasing the mip level by the specified value.", "Minimum shader model": 1},
    "tex3Dgrad": {"Description": "Samples a 3D texture using a gradient to select the mip level.", "Minimum shader model": 1},
    "tex3Dlod": {"Description": "Samples a 3D texture at a specified mip level.", "Minimum shader model": 1},
    "tex3Dproj": {"Description": "Samples a 3D texture and projects the texture coordinate by dividing it by the last component.", "Minimum shader model": 1},
    "texCUBE": {"Description": "Samples a texture.", "Minimum shader model": 1},
    "texCUBEbias": {"Description": "Samples a texture after biasing the mip level by the specified value.", "Minimum shader model": 1},
    "texCUBEgrad": {"Description": "Samples a texture using a gradient to select the mip level.", "Minimum shader model": 1},
    "texCUBElod": {"Description": "Samples a texture at a specified mip level.", "Minimum shader model": 1},
    "texCUBEproj": {"Description": "Samples a texture and projects the texture coordinate by dividing it by the last component.", "Minimum shader model": 1},
    "transpose": {"Description": "Transposes the specified input matrix.", "Minimum shader model": 1},
    "trunc": {"Description": "Returns the integer part of x; the fractional part is discarded.", "Minimum shader model": 1},
    "NonUniformResourceIndex": {"Description": "Ensures non-uniform indexing for resources like textures in shaders.", "Minimum shader model": 6.0},
    "AddUint64": {"Description": "Unsigned add of 32-bit operand with the carry.", "Minimum shader model": 6.0},
    "GetAttributeAtVertex": {"Description": "Returns the values of the attributes at the vertex.", "Minimum shader model": 6.1},
    "asfloat16": {"Description": "Converts integer bit patterns to half-precision floating-point values.", "Minimum shader model": 6.2},
    "asint16": {"Description": "Converts bit patterns to 16-bit integer values.", "Minimum shader model": 6.2},
    "asuint16": {"Description": "Converts bit patterns to 16-bit unsigned integer values.", "Minimum shader model": 6.2},
    "WaveGetLaneCount": {"Description": "Returns the number of lanes in the current wave.", "Minimum shader model": 6.0},
    "WaveGetLaneIndex": {"Description": "Returns the index of the current lane within its wave.", "Minimum shader model": 6.0},
    "WaveIsFirstLane": {"Description": "Checks if the current lane is the first lane in its wave.", "Minimum shader model": 6.0},
    "WaveActiveAnyTrue": {"Description": "Returns true if any lane in the wave evaluates to true.", "Minimum shader model": 6.0},
    "WaveActiveAllTrue": {"Description": "Returns true if all lanes in the wave evaluate to true.", "Minimum shader model": 6.0},
    "WaveActiveBallot": {"Description": "Returns a bitmask where each bit represents whether the condition is true for each lane.", "Minimum shader model": 6.0},
    "WaveReadLaneAt": {"Description": "Reads the value of a specific lane within the wave using an index.", "Minimum shader model": 6.0},
    "WaveReadLaneFirst": {"Description": "Reads the value from the first lane in the current wave.", "Minimum shader model": 6.0},
    "WaveActiveAllEqual": {"Description": "Checks if all lanes in the wave have equal values.", "Minimum shader model": 6.0},
    "WaveActiveBitAnd": {"Description": "Performs a bitwise AND operation across all lanes in the wave and returns the result.", "Minimum shader model": 6.0},
    "WaveActiveBitOr": {"Description": "Performs a bitwise OR operation across all lanes in the wave and returns the result.", "Minimum shader model": 6.0},
    "WaveActiveBitXor": {"Description": "Performs a bitwise XOR operation across all lanes in the wave and returns the result.", "Minimum shader model": 6.0},
    "WaveActiveCountBits": {"Description": "Counts the number of active lanes (lanes where the condition is true) in the wave.", "Minimum shader model": 6.0},
    "WaveActiveMax": {"Description": "Returns the maximum value among all active lanes in the wave.", "Minimum shader model": 6.0},
    "WaveActiveMin": {"Description": "Returns the minimum value among all active lanes in the wave.", "Minimum shader model": 6.0},
    "WaveActiveProduct": {"Description": "Computes the product of values across all active lanes in the wave.", "Minimum shader model": 6.0},
    "WaveActiveSum": {"Description": "Computes the sum of values across all active lanes in the wave.", "Minimum shader model": 6.0},
    "WavePrefixCountBits": {"Description": "Computes the prefix count of active bits up to the current lane in the wave.", "Minimum shader model": 6.0},
    "WavePrefixSum": {"Description": "Computes the prefix sum up to the current lane in the wave.", "Minimum shader model": 6.0},
    "WavePrefixProduct": {"Description": "Computes the prefix product up to the current lane in the wave.", "Minimum shader model": 6.0},
    "QuadReadLaneAt": {"Description": "Reads the value of a specific lane within the quad using an index.", "Minimum shader model": 6.0},
    "QuadReadAcrossDiagonal": {"Description": "Reads values across the diagonal of a quad group.", "Minimum shader model": 6.0},
    "QuadReadAcrossX": {"Description": "Reads values across the X axis of a quad group.", "Minimum shader model": 6.0},
    "QuadReadAcrossY": {"Description": "Reads values across the Y axis of a quad group.", "Minimum shader model": 6.0},
    "dot4add_u8packed": {"Description": "Unsigned dot product of 4 x u8 vectors packed into i32, with accumulate to i32.", "Minimum shader model": 6.4},
    "dot4add_i8packed": {"Description": "Signed dot product of 4 x i8 vectors packed into i32, with accumulate to i32.", "Minimum shader model": 6.4},
    "dot2add": {"Description": "2D half dot product with accumulate to float.", "Minimum shader model": 6.4},
    "WaveMultiPrefixCountBits": {"Description": "Returns the count of bits set to 1 on groups of lanes identified by a bitmask.", "Minimum shader model": 6.5},
    "WaveMultiPrefixProduct": {"Description": "Returns the result of the operation on groups of lanes identified by a bitmask.", "Minimum shader model": 6.5},
    "WaveMatch": {"Description": "Checks if all lanes in the wave have the same value.", "Minimum shader model": 6.5},
    "WaveMultiPrefixBitAnd": {"Description": "Performs a multi-wave prefix AND operation across all lanes in the wave and returns the result.", "Minimum shader model": 6.5},
    "WaveMultiPrefixBitOr": {"Description": "Performs a multi-wave prefix OR operation across all lanes in the wave and returns the result.", "Minimum shader model": 6.5},
    "WaveMultiPrefixBitXor": {"Description": "Performs a multi-wave prefix XOR operation across all lanes in the wave and returns the result.", "Minimum shader model": 6.5},
    "WaveMultiPrefixSum": {"Description": "Performs a multi-wave prefix sum operation across all lanes in the wave and returns the result.", "Minimum shader model": 6.5},
    "IsHelperLane": {"Description": "Returns true on helper lanes in pixel shaders.", "Minimum shader model": 6.6},
    "QuadAny": {"Description": "Compares boolean across a quad.", "Minimum shader model": 6.7},
    "QuadAll": {"Description": "Compares boolean across a quad.", "Minimum shader model": 6.7},
    "TraceRay": {"Description": "Initiates ray tracing, allowing intersection testing and tracing rays into the scene geometry.", "Minimum shader model": 6.6},
    "ReportHit": {"Description": "Used to report a hit when a ray intersects with geometry to record info about the intersection.", "Minimum shader model": 6.6},
    "CallShader": {"Description": "Used to invoke a function or a callable shader, allowing shaders to be modular and reusable within a ray tracing pipeline.", "Minimum shader model": 6.6},
    "IgnoreHit": {"Description": "Used to skip uneeded calculations by indicating a hit should be ignored or not processed further.", "Minimum shader model": 6.6},
    "AcceptHitAndEndSearch": {"Description": "Used to indicate that a hit has been accepted and to terminate further intersection testing.", "Minimum shader model": 6.6},
    "DispatchRaysIndex": {"Description": "Used to dispatch rays with an index indicating the starting point for ray dispatch.", "Minimum shader model": 6.6},
    "DispatchRaysDimensions": {"Description": "Used to dispatch rays specifying the dimensions (width and height) for the ray dispatch.", "Minimum shader model": 6.6},
    "WorldRayOrigin": {"Description": "Represents the origin point of a ray in world coordinates.", "Minimum shader model": 6.6},
    "WorldRayDirection": {"Description": "Represents the direction vector of a ray in world coordinates.", "Minimum shader model": 6.6},
    "ObjectRayOrigin": {"Description": "Origin of a ray in object-local coordinates, used in ray tracing shaders.", "Minimum shader model": 6.6},
    "ObjectRayDirection": {"Description": "Direction of a ray in object-local coordinates, used in ray tracing shaders.", "Minimum shader model": 6.6},
    "RayTMin": {"Description": "Minimum t-value along a ray's direction, used in ray tracing shaders.", "Minimum shader model": 6.6},
    "RayTCurrent": {"Description": "Current t-value along a ray's direction, used in ray tracing shaders.", "Minimum shader model": 6.6},
    "PrimitiveIndex": {"Description": "Index of the primitive being processed, used in geometry shaders.", "Minimum shader model": 6.6},
    "InstanceID": {"Description": "ID of the instance being processed, used in instanced rendering.", "Minimum shader model": 6.6},
    "InstanceIndex": {"Description": "Index of the instance being processed, used in instanced rendering.", "Minimum shader model": 6.6},
    "GeometryIndex": {"Description": "Index of the geometry being processed, used in geometry shaders.", "Minimum shader model": 6.6},
    "HitKind": {"Description": "Hit result reported in a ray tracing shader, indicating what type of intersection was found.", "Minimum shader model": 6.6},
    "RayFlags": {"Description": "Represents flags associated with a ray, used to convey how the ray should be processed.", "Minimum shader model": 6.6},
    "ObjectToWorld3x4": {"Description": "Represents a 3x4 matrix used to transform coordinates from object-local space to world space.", "Minimum shader model": 6.6},
    "WorldToObject3x4": {"Description": "Represents a 3x4 matrix used to transform coordinates from world space to object-local space.", "Minimum shader model": 6.6},
    "ObjectToWorld4x3": {"Description": "Represents a 4x3 matrix used to transform coordinates from object-local space to world space.", "Minimum shader model": 6.6},
    "WorldToObject4x3": {"Description": "Represents a 4x3 matrix used to transform coordinates from world space to object-local space.", "Minimum shader model": 6.6},
    "unpack_s8s16": {"Description": "Unpacks a signed 8-bit value into a signed 16-bit value.", "Minimum shader model": 6.6},
    "unpack_u8u16": {"Description": "Unpacks an unsigned 8-bit value into an unsigned 16-bit value.", "Minimum shader model": 6.6},
    "unpack_s8s32": {"Description": "Unpacks a signed 8-bit value into a signed 32-bit value.", "Minimum shader model": 6.6},
    "unpack_u8u32": {"Description": "Unpacks an unsigned 8-bit value into an unsigned 32-bit value.", "Minimum shader model": 6.6},
    "pack_s8": {"Description": "Packs a signed 8-bit value.", "Minimum shader model": 6.6},
    "pack_u8": {"Description": "Packs an unsigned 8-bit value.", "Minimum shader model": 6.6},
    "pack_clamp_s8": {"Description": "Packs and clamps a signed 8-bit value.", "Minimum shader model": 6.6},
    "pack_clamp_u8": {"Description": "Packs and clamps an unsigned 8-bit value.", "Minimum shader model": 6.6},
    "SetMeshOutputCounts": {"Description": "Is used to set the output counts for different types of mesh primitives in geometry shaders.", "Minimum shader model": 6.5},
    "DispatchMesh": {"Description": "Is used to dispatch mesh shader threads for processing mesh primitives.", "Minimum shader model": 6.5},
    "AllocateRayQuery": {"Description": "Allocates resources for ray queries in ray tracing shaders.", "Minimum shader model": 6.5},
    "CreateResourceFromHeap": {"Description": "Creates a resource from a heap, typically used for dynamic resource allocation.", "Minimum shader model": 6.6},
    "and": {"Description": "", "Minimum shader model": ""},
    "or": {"Description": "", "Minimum shader model": ""},
    "select": {"Description": "A conditional selection function that chooses between two values based on a condition.", "Minimum shader model": ""},
    "Barrier": {"Description": "Request a barrier for a set of memory types and/or thread group execution sync.", "Minimum shader model": 6.8},
    "GetRemainingRecursionLevels": {"Description": "Returns how many levels of recursion remain.", "Minimum shader model": 6.8},
}



remarks_base = {'NonUniformResourceIndex' : 'See [NonUniformResourceIndex semantics](https://microsoft.github.io/DirectX-Specs/d3d/WorkGraphs#nonuniformresourceindex-semantics) and [**Resource Binding**](../direct3d12/resource-binding-in-hlsl.md).', 
                'GetAttributeAtVertex': '''
                    Attributes used with `GetAttributeAtVertex`
                    should be placed in the attribute structure for [`vertices`](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader#vertex-attributes),
                    and marked with the `nointerpolation` modifier.
                    In the case that there is a pixel shader input aligned with a mesh shader per-primitive output,
                    and that attribute is not marked as nointerpolation, the driver will still force the attribute to nointerpolate
                    in order to maintain functionality.\n
                    Vertex order is determined by the order of the vertex [`indices`](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader#vertex-indices) for the primitive.
                    The first vertex is the one referenced by the first index in this vector.
                    When the term *provoking vertex* is used in other feature descriptions,
                    for the mesh shader pipeline, it means the first vertex.
                    This order applies to the component order of `SV_Barycentrics`
                    and the index passed to `GetAttributeAtVertex`.''',
                'asfloat16' : 'See https://github.com/microsoft/DirectXShaderCompiler/wiki/16-Bit-Scalar-Types',
                'asint16'   : 'See https://github.com/microsoft/DirectXShaderCompiler/wiki/16-Bit-Scalar-Types',
                'asuint16'  : 'See https://github.com/microsoft/DirectXShaderCompiler/wiki/16-Bit-Scalar-Types',
                'AddUint64' : '`AddUint64` is useful for high-precision computations where 64-bit integers are required.',   
                'WaveMultiPrefixCountBits' : 'See [WaveMultiPrefix*() Functions](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_5#wavemultiprefix-functions)',
                'WaveMultiPrefixProduct'   : 'See [WaveMultiPrefix*() Functions](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_5#wavemultiprefix-functions)',
                'WaveMatch'                : 'See [WaveMultiPrefix*() Functions](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_5#wavemultiprefix-functions)',
                'WaveMultiPrefixBitAnd'    : 'See [WaveMultiPrefix*() Functions](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_5#wavemultiprefix-functions)',
                'WaveMultiPrefixBitOr'     : 'See [WaveMultiPrefix*() Functions](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_5#wavemultiprefix-functions)',
                'WaveMultiPrefixBitXor'    : 'See [WaveMultiPrefix*() Functions](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_5#wavemultiprefix-functions)',
                'WaveMultiPrefixSum'       : 'See [WaveMultiPrefix*() Functions](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_5#wavemultiprefix-functions)',
                'IsHelperLane' :  'See [IsHelperLane](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_6#is-helper-lane)'
                }

component_type_doc_dict = {'float_like<>' : '[**float**](/windows/desktop/WinProg/windows-data-types) or [**double**](/windows/desktop/WinProg/windows-data-types)',
                       'float16_t<>'  : '[**half**](https://github.com/microsoft/DirectXShaderCompiler/wiki/16-Bit-Scalar-Types)',
                       'numeric<>' : '[**float**](/windows/desktop/WinProg/windows-data-types), [**int**](/windows/desktop/WinProg/windows-data-types)',
                       'uint' : '[**uint**](/windows/desktop/WinProg/windows-data-types)',
                       'uint<4>' : '[**uint**](/windows/desktop/WinProg/windows-data-types)',
                      }
size_dict =  {'float_like<>' : 'any', 'float16_t<>' : 'any', 'numeric<>'  : 'any', 
              'uint' : '1', 'uint<4>' : '4', }

type_order_dict = {'float_like<>' : '[**scalar**](dx-graphics-hlsl-intrinsic-scalar.md), **vector**, or **matrix**',
                   'float16_t<>' : '[**scalar**](dx-graphics-hlsl-intrinsic-scalar.md), **vector**, or **matrix**',
                   'numeric<>'  : '[**scalar**](dx-graphics-hlsl-intrinsic-scalar.md), **vector**, or **matrix**',
                   'uint'  : '[**scalar**](dx-graphics-hlsl-intrinsic-functions.md)',
                   'uint<4>' : '**vector**',
                  }

shader_model_dict = {'6.0' : '[Shader Model 6](shader-model-6-0.md)', 
                     '6.1' : 'Shader Model 6.1', 
                     '6.2' : 'Shader Model 6.2', 
                     '6.3' : 'Shader Model 6.3', 
                     '6.4' : '[Shader Model 6.4](hlsl-shader-model-6-4-features-for-direct3d-12.md)', 
                     '6.5' : '[Shader Model 6.5](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_5)', 
                     '6.6' : '[Shader Model 6.6](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_6)', 
                     '6.7' : '[Shader Model 6.7](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_7)',
                     '6.8' : '[Shader Model 6.8](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_8)',}


shader_stage_dict = {
'amplification' : 'https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html#amplification-shader-and-mesh-shader',
'anyhit'        : '../direct3d12/any-hit-shader.md',
'callable'      : '../direct3d12/callable-shader.md',
'closesthit'    : '../direct3d12/closest-hit-shader.md',
'compute'       : '../direct3d11/direct3d-11-advanced-stages-compute-shader.md',
'domain'        : 'https://learn.microsoft.com/en-us/windows/uwp/graphics-concepts/domain-shader-stage--ds-',
'geometry'      : 'https://learn.microsoft.com/en-us/windows/uwp/graphics-concepts/geometry-shader-stage--gs-',
'hull'          : 'https://learn.microsoft.com/en-us/windows/uwp/graphics-concepts/hull-shader-stage--hs-',
'intersection'  : '../direct3d12/intersection-shader.md',
'mesh'          : 'https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html',
'miss'          : '../direct3d12/miss-shader.md',
'raygeneration' : '../direct3d12/ray-generation-shader.md',
'pixel'         : 'dx-graphics-hlsl-writing-shaders-9.md#pixel-shader-basics',
'vertex'        : 'dx-graphics-hlsl-writing-shaders-9.md#vertex-shader-basics',
}

def gen_metadata(hl_op_name, short_description):
    ret_str =  f'---\ndescription: {short_description}\nnms.assetid:\ntitle: {hl_op_name}\n'                              
    ret_str += f'ms.topic: reference\nms.date: {datetime.today().strftime('%m/%d/%Y')}\n'
    ret_str += f'topic_type:\n- APIRef\n- kbSyntax\napi_name:\n- {hl_op_name}\napi_type:\n- NA\n---\n\n'
    return ret_str

def gen_header(hl_op_name):
    short_description = shader_functions[hl_op_name]["Description"]
    meta_data = gen_metadata(hl_func_name,short_description) + '\n'                                         
    return f'{meta_data}# {hl_op_name}\n\n{short_description}\n\n'


def gen_syntax(hl_op_name, params):
    ret_str = '## Syntax\n\n\n'
    ret_str += '```syntax\n'
    func_signature = f'{params["ret"]} {hl_op_name}('
    for name, type_str in params.items():
        if name == 'ret':
             continue
        func_signature += f'{type_str} {name}, '
    func_signature = func_signature.rstrip(", ") + ");\n"
    ret_str += f'{func_signature}```\n\n'
    return ret_str


def gen_params(params):
    ret_params = f'## Parameters\n\n{"This function has no parameters.\n\n" if len(params) == 0 else ""}'
    for name, (type_str, description) in params.items():
        ret_params +=f'''<dl> <dt>\n\n
        *{name}* : {type_str} \n</dt><dd>\n\n
        {description}</dd></dl>\n\n'''
    return ret_params

def gen_return(description):
    return f'## Return value\n\n {description}'

def gen_type_description(params):
    ret_str = '## Type Description\n\n'
    ret_str += '| Name  | [**Template Type**](dx-graphics-hlsl-intrinsic-functions.md)| [**Component Type**](dx-graphics-hlsl-intrinsic-functions.md) | Size |\n'
    ret_str += '|-------|-------------------------------------------------------------|---------------------------------------------------------------|------|\n'

    for name, type_str in params.items():
        ret_str += f'| *{name}*   | {type_order_dict.get(type_str, type_str)} | {component_type_doc_dict.get(type_str, type_str)} | {size_dict.get(type_str, type_str)} |\n'
    return ret_str

def gen_remarks(long_desc):
    return f'## Remarks\n\n{long_desc}'


def gen_min_shader_model(hl_op_name, hlsl_to_dxil_op, dxil_op_to_docs):
    ret_str = '## Minimum Shader Model\n\n'
    ret_str += 'This function is supported in the following shader models.\n'
    ret_str += '|Shader Model |	Supported|\n'
    ret_str += '|-------------|----------|\n'
    dxil_op = hlsl_to_dxil_op.get(hl_op_name,-1)
    if dxil_op != -1:
        dxil_docs = dxil_op_to_docs[dxil_op]
        ret_str += f'|{shader_model_dict[dxil_docs[1]]} and higher shader models | yes |\n'
    return ret_str

def gen_shader_stages(hl_op_name, hlsl_to_dxil_op, dxil_op_to_docs):
    ret_str = '## Shader Stages\n\n'
    dxil_op = hlsl_to_dxil_op.get(hl_op_name,-1)
    if dxil_op != -1:
        dxil_docs = dxil_op_to_docs[dxil_op]
        for sm in dxil_docs[3]:
            url = shader_stage_dict.get(sm,'')
            name = f'**{sm[0].upper()}{sm[1:]} Shader**'
            if url != '':
                ret_str += f'* [{name}]({url})\n'
            else:
                ret_str += f'* {name}\n'                         
    return ret_str

if __name__ == "__main__":
    keys = ApiKeys.parse_api_keys()
    OPENAI_API_KEY = keys.open_ai_key
    hlsl_to_dxil_op = load_dict("hlsl_intrinsics.pkl", run_dxc)
    dxil_op_to_docs = query_dxil()

    hl_func_params = get_intrinsic_param_types(undocumented_apis)
    #md_file = gen_header(hl_func_name)
    #md_file += gen_syntax(hl_func_name, params)
    #md_file += gen_type_description(params) + '\n'
    #md_file += gen_min_shader_model(hl_func_name, hlsl_to_dxil_op, dxil_op_to_docs) + '\n'
    #md_file += gen_shader_stages(hl_func_name, hlsl_to_dxil_op, dxil_op_to_docs) + '\n'
    #print(md_file)
    for hl_func_name, params in hl_func_params.items():
        print(hl_func_name)
        documentation = document_hlsl_intrinsic(hl_func_name, params)
        print(documentation)