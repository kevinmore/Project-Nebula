#pragma once
#include <Utility/DataTypes.h>
#include <assert.h>
#include <QtGui/QQuaternion>
#include <assimp/scene.h>
#include <Utility/Math.h>

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

	mat4 m_globalNodeTransform;

	bool isXConstraint;

	/** Parent bode. NULL if this node is the root. **/
	Bone* m_parent;

	Bone() 
	{ 
		m_parent = NULL;
		isXConstraint = false;
	}

	Bone(Bone* parent)
	{
		// init the current bone
		m_parent = parent;

		// add the current bone to its parent if it's not the root
		if(parent) parent->addChild(this);

		isXConstraint = false;
	}


	void calcWorldTransform()
	{
// 		// The bone's transformation in the skeleton space,
// 		// aka the bind matrix - the bone's parent's local matrices concatenated with the bone's local matrix.
// 		const aiMatrix4x4 nodeGlobalTransform = Math::convToAiMat4(m_finalTransform);
// 		// The transformation relative to the bone's parent (parent => bone space).
// 		aiMatrix4x4 nodeLocalTransform;
// 		if (m_parent && m_parent->m_name != "Project Nebula Skeleton ROOT")
// 		{
// 			aiMatrix4x4 parentGlobalTransform = Math::convToAiMat4(m_parent->m_finalTransform);
// 			aiMatrix4x4 inverseParentGlobalTransform = parentGlobalTransform.Inverse();
// 			nodeLocalTransform = inverseParentGlobalTransform * nodeGlobalTransform ; // N = P^-1 * B
// 		}
// 		else
// 		{
// 			nodeLocalTransform = Math::convToAiMat4(m_nodeTransform);
// 		}
// 
// 		aiMatrix4x4 globalTransform = nodeLocalTransform * Math::convToAiMat4(m_offsetMatrix);

		aiMatrix4x4 globalTransform = Math::convToAiMat4(m_globalNodeTransform);

		aiVector3D	 scaling;
		aiQuaternion rotation;
		aiVector3D	 position;
		globalTransform.Decompose(scaling, rotation, position);
		
		m_worldPos = vec3(position.x, position.y, position.z);
		m_worldScaling = vec3(scaling.x, scaling.y, scaling.z);
		m_worldQuaternion = QQuaternion(rotation.w, rotation.x, rotation.y, rotation.z);
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
		else return m_children[i];
	}

	int childCount()
	{
		return m_children.size();
	}

	vec3 getWorldPosition()
	{
		return m_worldPos;
	}

	QQuaternion getWorldRotation()
	{
		return m_worldQuaternion;
	}

	void setWorldRotation(const QQuaternion& roation)
	{
		m_nodeTransform.rotate(roation);

		m_worldQuaternion = roation;
	}

	void setWorldPosition(const vec3 &newPos)
	{
		vec3 originalPos = m_worldPos;
		vec3 deltaTranslation = newPos - originalPos;

		// here the bone is translated in the world coordinates
		// dont use QMatrix4x4.translate(), that only translates in the relative coordinates
		m_globalNodeTransform(0, 3) += deltaTranslation.x();
		m_globalNodeTransform(1, 3) += deltaTranslation.y();
		m_globalNodeTransform(2, 3) += deltaTranslation.z();

		aiMatrix4x4 parentGlobalTransform = Math::convToAiMat4(m_parent->m_globalNodeTransform);
		aiMatrix4x4 inverseParentGlobalTransform = parentGlobalTransform.Inverse();

		m_nodeTransform = Math::convToQMat4(&inverseParentGlobalTransform) * m_globalNodeTransform;
		calcWorldTransform();
	}

private:
	/** Position, rotation, scaling. **/
	vec3 m_worldPos;
	vec3 m_worldScaling;
	QQuaternion m_worldQuaternion;

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