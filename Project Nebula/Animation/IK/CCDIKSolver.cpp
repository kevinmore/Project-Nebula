#include "CCDIKSolver.h"


CCDIKSolver::CCDIKSolver( uint iterations )
	: m_iterations( iterations )
{}



CCDIKSolver::~CCDIKSolver(void)
{}

bool CCDIKSolver::solve( const QVector<IkConstraint>& constraints, Skeleton* skeleton )
{

	// indicates whether a chain set failed to be solved correctly
	bool allSolved = true;

	// how many constraint chain sets to solve for?
	const int numConstraintSets = constraints.size();

	// solve for each constraint chain set
	for( int i = 0; i < numConstraintSets; i++ )
	{
		const CCDIKSolver::IkConstraint& constraint = constraints[i];

		allSolved = solveOneConstraint(constraint, skeleton);

	}

	// solved correctly?
	return allSolved;
}

bool CCDIKSolver::solveOneConstraint( const IkConstraint& constraint, Skeleton* skeleton )
{
	Bone* effectorBone = constraint.m_endBone;
	Bone* baseBone = constraint.m_startBone;

	// find the set of bones within this chain
	QVector<Bone*> boneChain;
	skeleton->getBoneChain(baseBone, effectorBone, boneChain);

	// if there are bones in the chain
	if (!boneChain.size()) return false;

	QVector<float> m_distances;
	float m_totalChainLength;

	for(int i = 0; i < boneChain.size() - 1; ++i)
		m_distances.push_back(skeleton->getDistanceBetween(boneChain[i], boneChain[i + 1]));

	m_totalChainLength = 0.0f;
	for (int i = 0; i < m_distances.size(); ++i)
	{
		m_totalChainLength += m_distances[i];
	}

	float rootToTargetLenght = (constraint.m_targetMS - baseBone->getWorldPosition()).length();
	if(m_totalChainLength - rootToTargetLenght < 0.00001f)
	{
	//	qDebug() << "Target out of range.";
		return false;
	}
	
	// begin the iteration
	for( uint iteration = 0; iteration < m_iterations; ++iteration )
	{
		// check if the target is already reached
		if (qFuzzyIsNull((effectorBone->getWorldPosition() - constraint.m_targetMS).length()))
		{
			qDebug() << "Target reached.";
			return true;
		}

		// skip the effector
		for( int jointIndex = boneChain.size() - 2; jointIndex >= 0; --jointIndex )
		{
			Bone* joint = boneChain[jointIndex];


			const vec3 effectorPos = effectorBone->getWorldPosition();
			const vec3 jointPos = joint->getWorldPosition();

			// joint to effector bone direction
			vec3 localJoint2End = (effectorPos - jointPos).normalized();

			// joint to target direction
			vec3 localJoint2Target = (constraint.m_targetMS - jointPos).normalized();

			// calculate the rotation axis and angle
			const vec3 rotationAxis = vec3::crossProduct(localJoint2End, localJoint2Target);
			float deltaAngle = qRadiansToDegrees(qAcos(vec3::dotProduct(localJoint2End,  localJoint2Target)));

			// if the angle is too small
			if (qFuzzyIsNull(deltaAngle))
			{  
				continue;
			}

			// limit the angle
			deltaAngle = qBound(-100.0f, deltaAngle, 100.0f);

			QQuaternion deltaRotation = QQuaternion::fromAxisAndAngle(rotationAxis, -deltaAngle);

			//check the DOF of the joint
			if (joint->isXConstraint)
			{
				float curYaw, curPitch, curRoll;  
				float deltaYaw, deltaPitch, deltaRoll;  


				if (iteration == 0)
				{  
					deltaRotation = QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_Y, -deltaAngle);  
				} 
				else  
				{  
					vec3 eulerAngles = Math::QuaternionToEuler(deltaRotation);
					deltaYaw = eulerAngles.z();
					deltaPitch = eulerAngles.y();
					deltaRoll = eulerAngles.x();

					eulerAngles = Math::QuaternionToEuler(joint->getWorldRotation());
					curYaw = eulerAngles.z();
					curPitch = eulerAngles.y();
					curRoll = eulerAngles.x();

					if (qFuzzyIsNull(deltaPitch))
					{  
						continue;
					}  

					// limit the yaw [-0.002f - curYaw, M_PI - curYaw]  
					deltaPitch = qBound(-0.002f - curPitch, deltaPitch, float( M_PI ) - curPitch);
					deltaPitch = qBound(qDegreesToRadians(-100.0f), deltaPitch, qDegreesToRadians(100.0f));

					deltaRotation = Math::QuaternionFromEuler(vec3(0.0f, -deltaPitch, 0.0f));
				}  
			}
			
			joint->m_nodeTransform.rotate(deltaRotation);
			joint->m_globalNodeTransform.rotate(deltaRotation);
			aiMatrix4x4 parentGlobalTransform = Math::convToAiMat4(joint->m_parent->m_globalNodeTransform);
			aiMatrix4x4 inverseParentGlobalTransform = parentGlobalTransform.Inverse();
			joint->m_nodeTransform = Math::convToQMat4(&inverseParentGlobalTransform) * joint->m_globalNodeTransform;

			// re-sort the skeleton pose
			skeleton->sortPose(baseBone, baseBone->m_parent->m_globalNodeTransform);
		}
						
	}

	return true;
	
}


void CCDIKSolver::BoneTransform( Skeleton* skeleton, Bone* baseBone, QVector<mat4>& Transforms )
{
	QVector<Bone*> boneList;
	skeleton->makeBoneListFrom(baseBone, boneList);
	Transforms.resize(skeleton->getSkeletonSize());
	for (int i = 0; i < boneList.size(); ++i)
	{
		Transforms[boneList[i]->m_ID] = boneList[i]->m_finalTransform;
	}
}
