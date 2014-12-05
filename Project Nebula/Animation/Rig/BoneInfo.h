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
	// bind pose matrix
	mat4 m_offsetMatrix;

	// the bone's transform in bone space
	mat4 m_boneSpaceTransform;

	// the bone's transform in model space
	mat4 m_modelSpaceTransform;

	// final matrix for the shader
	mat4 m_finalTransform;

	/** Parent bode. NULL if this node is the root. **/
	Bone* m_parent;

	/** Member functions **/

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

	/** Utility functions **/

	// this function decompose the node transform matrix in model space into
	// 3 components: scaling, rotation, position
	void decomposeModelSpaceTransform()
	{
		Math::decomposeMat4(m_modelSpaceTransform, m_modelSpaceScaling, m_modelSpaceRotation, m_modelSpacePosition);
	}

	vec3 getModelSpacePosition()
	{
		return m_modelSpacePosition;
	}

	QQuaternion getModelSpaceRotation()
	{
		return m_modelSpaceRotation;
	}

	vec3 getModelSpaceScaling()
	{
		return m_modelSpaceScaling;
	}

	void rotateInModelSpace(const QQuaternion& deltaRoation)
	{
		m_modelSpaceTransform.rotate(deltaRoation);
		m_boneSpaceTransform = m_parent->m_modelSpaceTransform.inverted() * m_modelSpaceTransform;

		decomposeModelSpaceTransform();
	}


	void setModelSpacePosition(const vec3 &newPos)
	{
		vec3 originalPos = m_modelSpacePosition;
		vec3 deltaTranslation = newPos - originalPos;

		// clean the rotation updated by the FKController
		m_modelSpaceTransform.setToIdentity();
		//m_globalNodeTransform.translate(deltaTranslation);

		// here the bone is translated in the world coordinates
		// dont use QMatrix4x4.translate(), that only translates in the relative coordinates
		m_modelSpaceTransform(0, 3) += deltaTranslation.x();
		m_modelSpaceTransform(1, 3) += deltaTranslation.y();
		m_modelSpaceTransform(2, 3) += deltaTranslation.z();


		m_boneSpaceTransform = m_parent->m_modelSpaceTransform.inverted() * m_modelSpaceTransform;
		decomposeModelSpaceTransform();
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

	// returns the euler angles in degrees
	Math::EulerAngle getEulerAnglesInBoneSpace()
	{
		vec3 pos, scale;
		QQuaternion rot;
		Math::decomposeMat4(m_boneSpaceTransform, scale, rot, scale);

		Math::EulerAngle ea = Math::QuaternionToEuler(rot);
		return Math::EulerAngle(qRadiansToDegrees(ea.m_fRoll), 
			                    qRadiansToDegrees(ea.m_fPitch),
			                    qRadiansToDegrees(ea.m_fYaw));
	}

	// returns the euler angles in degrees
	Math::EulerAngle getEulerAnglesInModelSpace()
	{
		vec3 pos, scale;
		QQuaternion rot;
		Math::decomposeMat4(m_modelSpaceTransform, scale, rot, scale);

		Math::EulerAngle ea = Math::QuaternionToEuler(rot);
		return Math::EulerAngle(qRadiansToDegrees(ea.m_fRoll), 
								qRadiansToDegrees(ea.m_fPitch),
								qRadiansToDegrees(ea.m_fYaw));
	}

private:
	DimensionOfFreedom m_DOF;


	/** Position, rotation, scaling in model space. **/
	vec3 m_modelSpacePosition;
	vec3 m_modelSpaceScaling;
	QQuaternion m_modelSpaceRotation;

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