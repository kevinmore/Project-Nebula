#pragma once
#include <Utility/DataTypes.h>
#define NUM_BONES_PER_VEREX 4

struct VertexBoneData
{ 
	uint IDs[NUM_BONES_PER_VEREX];
	float Weights[NUM_BONES_PER_VEREX];

	VertexBoneData()
	{
		reset();
	}

	void reset()
	{
		ZERO_MEM(IDs);
		ZERO_MEM(Weights);        
	}

	void addBoneData(uint boneID, float weight)
	{
		for (uint i = 0 ; i < ARRAY_SIZE_IN_ELEMENTS(IDs) ; ++i) 
		{
			if (Weights[i] == 0.0) 
			{
				IDs[i]     = boneID;
				Weights[i] = weight;
				return;
			}        
		}
	}
};

struct BoneInfo
{
	mat4 boneOffset;
	mat4 finalTransformation;        

	BoneInfo(){}
};

struct Vertex
{
	vec3 postition;
	vec4 color;
	vec3 normal;
	vec2 texCoord;
	vec4 boneIDs;
	vec4 boneWeights;
};

