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


	/** Parent bode. NULL if this node is the root. **/
	Bone* m_parent;

	Bone() 
	{ 
		m_parent = NULL;
		m_DOF = DimensionOfFreedom();
	}

	Bone(Bone* parent)
	{
		// init the current bone
		m_parent = parent;

		// add the current bone to its parent if it's not the root
		if(parent) parent->addChild(this);

		m_DOF = DimensionOfFreedom();
	}


	void calcWorldTransform()
	{
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

	void rotateInWorldSpace(const QQuaternion& deltaRoation)
	{
		m_globalNodeTransform.rotate(deltaRoation);
		m_nodeTransform = m_parent->m_globalNodeTransform.inverted() * m_globalNodeTransform;

		calcWorldTransform();
	}


	void setWorldPosition(const vec3 &newPos)
	{
		vec3 originalPos = m_worldPos;
		vec3 deltaTranslation = newPos - originalPos;

		// clean the rotation updated by the FKController
		m_globalNodeTransform.setToIdentity();
		//m_globalNodeTransform.translate(deltaTranslation);

		// here the bone is translated in the world coordinates
		// dont use QMatrix4x4.translate(), that only translates in the relative coordinates
		m_globalNodeTransform(0, 3) += deltaTranslation.x();
		m_globalNodeTransform(1, 3) += deltaTranslation.y();
		m_globalNodeTransform(2, 3) += deltaTranslation.z();


		m_nodeTransform = m_parent->m_globalNodeTransform.inverted() * m_globalNodeTransform;
		calcWorldTransform();
	}



	// DOF in degrees
	struct AngleLimits
	{
		float minAngle;
		float maxAngle;

		// default, totally free
		AngleLimits()
		{
			minAngle = -180.0f;
			maxAngle = 180.0f;
		}

		AngleLimits(float min, float max)
		{
			minAngle = min;
			maxAngle = max;
		}
	};

	// DOF in degrees
	struct DimensionOfFreedom
	{
		AngleLimits PitchConstraint;
		AngleLimits YawConstraint;
		AngleLimits RollConstraint;

		DimensionOfFreedom()
		{
			PitchConstraint = AngleLimits();
			YawConstraint   = AngleLimits();
			RollConstraint  = AngleLimits();
		}

		DimensionOfFreedom(AngleLimits& pitchConstraint, AngleLimits& yawConstraint, AngleLimits& rollConstraint)
		{
			PitchConstraint = pitchConstraint;
			YawConstraint = yawConstraint;
			RollConstraint = rollConstraint;
		}
	};

	void setDof(DimensionOfFreedom& dof)
	{
		m_DOF = dof;
	}

	DimensionOfFreedom getDof()
	{
		return m_DOF;
	}

	uint getDofNumber()
	{
		uint result = 3;

		if (m_DOF.PitchConstraint.minAngle == m_DOF.PitchConstraint.maxAngle)
			--result;
		if (m_DOF.YawConstraint.minAngle == m_DOF.YawConstraint.maxAngle)
			--result;
		if (m_DOF.RollConstraint.minAngle == m_DOF.RollConstraint.maxAngle)
			--result;

		return result;
	}

	Math::EulerAngle getLocalEulerAngleInDegrees()
	{
		aiMatrix4x4 localTransform = Math::convToAiMat4(m_nodeTransform);

		aiVector3D	 scaling;
		aiQuaternion rotation;
		aiVector3D	 position;
		localTransform.Decompose(scaling, rotation, position);
		QQuaternion localRotation = QQuaternion(rotation.w, rotation.x, rotation.y, rotation.z);

		Math::EulerAngle ea = Math::QuaternionToEuler(localRotation);
		return Math::EulerAngle(qRadiansToDegrees(ea.m_fRoll), 
			                    qRadiansToDegrees(ea.m_fPitch),
			                    qRadiansToDegrees(ea.m_fYaw));
	}

	Math::EulerAngle getGlobalAngleInDegrees()
	{
		aiMatrix4x4 globalTransform = Math::convToAiMat4(m_globalNodeTransform);

		aiVector3D	 scaling;
		aiQuaternion rotation;
		aiVector3D	 position;
		globalTransform.Decompose(scaling, rotation, position);
		QQuaternion worldQuaternion = QQuaternion(rotation.w, rotation.x, rotation.y, rotation.z);

		Math::EulerAngle ea = Math::QuaternionToEuler(worldQuaternion);
		return Math::EulerAngle(qRadiansToDegrees(ea.m_fRoll), 
			qRadiansToDegrees(ea.m_fPitch),
			qRadiansToDegrees(ea.m_fYaw));
	}
private:
	DimensionOfFreedom m_DOF;


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