#pragma once
#include <Utility/DataTypes.h>
#include <assert.h>

class Bone
{
public:

	/** ID and Name. **/
	uint m_ID;
	QString m_name;

	/** Matrixes. **/
	mat4 m_offsetMatrix;
	mat4 m_nodeTransform;
	mat4 m_finalTransform;

	/** Parent bode. NULL if this node is the root. **/
	Bone* m_parent;

	Bone() { m_parent = NULL; }

	Bone(Bone* parent)
	{
		// init the current bone
		m_parent = parent;

		// add the current bone to its parent if it's not the root
		if(parent) parent->addChild(this);
	}

	mat4 getLocalTransformMatrix()
	{
		return m_nodeTransform * m_offsetMatrix;
	}

	void calcWorldPos()
	{
		float x = m_finalTransform(0, 3);
		float y = m_finalTransform(1, 3);
		float z = m_finalTransform(2, 3);
	
		m_worldPos = vec3(x, y, z);
	}

	void addChild(Bone* child)
	{
		m_children.push_back(child);
	}

	QVector<Bone*> getChildren()
	{
		return m_children;
	}

	Bone* getChild(uint i)
	{
		if(m_children.isEmpty()) return NULL;
		else return m_children.at(i);
	}

	int childCount()
	{
		return m_children.size();
	}

	vec3 getWorldPosition()
	{
		return m_worldPos;
	}

	void setWorldPosition(const vec3 &newPos)
	{
		vec3 originalPos = m_worldPos;
		vec3 delta = newPos - originalPos;

		// here the bone is translated in the world coordinates
		// dont use QMatrix4x4.translate(), that only translates in the relative coordinates
		m_finalTransform(0, 3) += delta.x();
		m_finalTransform(1, 3) += delta.y();
		m_finalTransform(2, 3) += delta.z();

		m_worldPos = newPos;
	}

private:
	/** Position. **/
	vec3 m_worldPos;

	/** The child bones of this bone. **/
	QVector<Bone*> m_children;
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