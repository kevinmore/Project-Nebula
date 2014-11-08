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

	mat4 m_worldMatrix;

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
		// The bone's transformation in the skeleton space,
		// aka the bind matrix - the bone's parent's local matrices concatenated with the bone's local matrix.
		const aiMatrix4x4 nodeGlobalTransform = Math::convToAiMat4(m_finalTransform);
		// The transformation relative to the bone's parent (parent => bone space).
		aiMatrix4x4 nodeLocalTransform;
		if (m_parent && m_parent->m_name != "Project Nebula Skeleton ROOT")
		{
			aiMatrix4x4 parentGlobalTransform = Math::convToAiMat4(m_parent->m_finalTransform);
			aiMatrix4x4 inverseParentGlobalTransform = parentGlobalTransform.Inverse();
			nodeLocalTransform = inverseParentGlobalTransform * nodeGlobalTransform ; // N = P^-1 * B
		}
		else
		{
			nodeLocalTransform = Math::convToAiMat4(m_nodeTransform);
		}

		aiMatrix4x4 globalTransform = nodeLocalTransform * Math::convToAiMat4(m_offsetMatrix);

		aiVector3D	 scaling;
		aiQuaternion rotation;
		aiVector3D	 position;
		globalTransform.Decompose(scaling, rotation, position);
		
		m_worldPos = vec3(position.x, position.y, position.z);
		m_worldScaling = vec3(scaling.x, scaling.y, scaling.z);
		m_worldQuaternion = QQuaternion(rotation.w, rotation.x, rotation.y, rotation.z);

		m_worldMatrix = Math::convToQMat4(&globalTransform);
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
// 		m_nodeTransform(0, 3) += delta.x();
// 		m_nodeTransform(1, 3) += delta.y();
// 		m_nodeTransform(2, 3) += delta.z();
// 
// 		m_worldMatrix(0, 3) = newPos.x();
// 		m_worldMatrix(1, 3) = newPos.y();
// 		m_worldMatrix(2, 3) = newPos.z();

// 		mat4 newTrans;
// 		newTrans.scale(m_worldScaling);
// 		newTrans.rotate(m_worldQuaternion);
// 		newTrans.translate(delta);
// 
// 		m_finalTransform = newTrans;
//		qDebug() << m_name << "m_deltaTranslation = " << m_deltaTranslation;
//		qDebug() << m_parent->m_name << "mparent->m_deltaTranslation = " << m_parent->m_deltaTranslation;
		m_nodeTransform.rotate(10, deltaTranslation);

		
		//calcWorldTransform();
		//m_offsetMatrix = Math::convToQMat4(&(Math::convToAiMat4(m_parent->m_finalTransform)* Math::convToAiMat4(m_worldMatrix)));

		//m_nodeTransform = m_parent->m_finalTransform * m_worldMatrix;

		m_worldPos = newPos;
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