#pragma once
#include <Utility/DataTypes.h>
#include <assert.h>
class BoneInfo
{
public:
	QString m_name;
	QMatrix4x4 m_localTransform;
	QMatrix4x4 m_globalTransform;        

	BoneInfo()
	{ 
		m_name = "";
		m_localTransform.fill(0);
		m_globalTransform.fill(0);
	}
};


#define NUM_BONES_PER_VEREX 4
struct VertexBoneData
{        
	uint IDs[NUM_BONES_PER_VEREX];
	float Weights[NUM_BONES_PER_VEREX];

	VertexBoneData()
	{
		Reset();
	};

	void Reset()
	{
		ZERO_MEM(IDs);
		ZERO_MEM(Weights);        
	}

	void AddBoneData(uint BoneID, float Weight)
	{
		for (uint i = 0 ; i < ARRAY_SIZE_IN_ELEMENTS(IDs) ; i++) {
			if (Weights[i] == 0.0) {
				IDs[i]     = BoneID;
				Weights[i] = Weight;
				return;
			}        
		}

		// should never get here - more bones than we have space for
		assert(0);
	}
};