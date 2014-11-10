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

		allSolved = solveOneConstraint(constraint, skeleton) && allSolved;

	}

	// solved correctly?
	return allSolved;
}

bool CCDIKSolver::solveOneConstraint( const IkConstraint& constraint, Skeleton* skeleton )
{
	Bone* effectorBone = constraint.m_endBone;
	Bone* baseBone = constraint.m_startBone;

	// re-sort the skeleton pose
	// this step is necessary because the parent of the baseBone might have moved
	skeleton->sortPose(baseBone, baseBone->m_parent->m_globalNodeTransform);

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
	if(m_totalChainLength - rootToTargetLenght < 0.1f)
	{
	//	qDebug() << "Target out of range.";
		return false;
	}
	
	// begin the iteration
	for( uint iteration = 0; iteration < m_iterations; ++iteration )
	{
		// check if the target is already reached
		if ((effectorBone->getWorldPosition() - constraint.m_targetMS).length() < 0.1f)
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
			vec3 currentDirection = (effectorPos - jointPos).normalized();

			// joint to target direction
			vec3 targetDirenction = (constraint.m_targetMS - jointPos).normalized();

			// calculate the rotation axis and angle
			const vec3 rotationAxis = vec3::crossProduct(currentDirection, targetDirenction);
			float deltaAngle = qRadiansToDegrees(qAcos(vec3::dotProduct(currentDirection,  targetDirenction)));
			// if the angle is too small
			if (deltaAngle < 0.1f || Math::isNaN(deltaAngle))
			{  
				continue;
				//return true;
			}
			deltaAngle *= -1; // right handed system
			Bone::DimensionOfFreedom dof = joint->getDof();
			QQuaternion curRotation, deltaRotation;
			Math::EulerAngle eulerAngles;
			float curYaw, curPitch, curRoll;  
			float deltaYaw, deltaPitch, deltaRoll;

			// Check DOF
			// get the current quaternion
			curRotation = joint->getWorldRotation();
			// decompose it
			eulerAngles = Math::QuaternionToEuler(deltaRotation);
			curRoll  = qRadiansToDegrees(eulerAngles.m_fRoll);
			curPitch = qRadiansToDegrees(eulerAngles.m_fPitch);
			curYaw   = qRadiansToDegrees(eulerAngles.m_fYaw);

			// create the delta quaternion
			deltaRotation = QQuaternion::fromAxisAndAngle(rotationAxis, deltaAngle);

			// decompose it
			eulerAngles = Math::QuaternionToEuler(deltaRotation);
			deltaRoll  = qRadiansToDegrees(eulerAngles.m_fRoll);
			deltaPitch = qRadiansToDegrees(eulerAngles.m_fPitch);
			deltaYaw   = qRadiansToDegrees(eulerAngles.m_fYaw);

			deltaRoll  = qBound(dof.Z_Axis_AngleLimits.minAngle, deltaRoll, dof.Z_Axis_AngleLimits.maxAngle);
			deltaPitch = qBound(dof.X_Axis_AngleLimits.minAngle, deltaPitch, dof.X_Axis_AngleLimits.maxAngle);
			deltaYaw   = qBound(dof.Y_Axis_AngleLimits.minAngle, deltaRoll, dof.Y_Axis_AngleLimits.maxAngle);

			// remake the quaternion
			deltaRotation = Math::QuaternionFromEuler(Math::EulerAngle(deltaRoll, deltaPitch, deltaYaw));

			
			eulerAngles = Math::QuaternionToEuler(deltaRotation);
			deltaRoll  = qRadiansToDegrees(eulerAngles.m_fRoll);
			deltaPitch = qRadiansToDegrees(eulerAngles.m_fPitch);
			deltaYaw   = qRadiansToDegrees(eulerAngles.m_fYaw);
			

			//check the DOF of the joint
// 			if (joint->isXConstraint)
// 			{
// 				if (iteration == 0)
// 				{  
// 					deltaRotation = QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_Y, -deltaAngle);  
// 				} 
// 				else  
// 				{  
// 					
// 
// 					vec3 eulerAngles = Math::QuaternionToEuler(deltaRotation);
// 					deltaYaw = eulerAngles.z();
// 					deltaPitch = eulerAngles.y();
// 					deltaRoll = eulerAngles.x();
// 
// 					eulerAngles = Math::QuaternionToEuler(joint->getWorldRotation());
// 					curYaw = eulerAngles.z();
// 					curPitch = eulerAngles.y();
// 					curRoll = eulerAngles.x();
// 
// 					if (qFuzzyIsNull(deltaPitch) || Math::isNaN(deltaPitch))
// 					{  
// 						continue;
// 					}  
// 
// 					deltaPitch = qBound(-0.002f - curPitch, deltaPitch, float( M_PI ) - curPitch);
// 					deltaPitch = qRadiansToDegrees(deltaPitch);
// 					deltaPitch = qBound(0.0f, deltaPitch, 100.0f);
// 
// 					deltaRotation = Math::QuaternionFromEuler(vec3(0.0f, -deltaPitch, 0.0f));
// 				}  
// 			}
			
			// adjust the world rotation of the joint
			joint->setWorldRotationDelta(deltaRotation);

			// re-sort the skeleton pose
			skeleton->sortPose(baseBone, baseBone->m_parent->m_globalNodeTransform);
		}
		
		m_effectorLastPos = effectorBone->getWorldPosition();
	}

	return true;
	
}


void CCDIKSolver::getBoneTransforms( Skeleton* skeleton, Bone* baseBone, QVector<mat4>& Transforms )
{
	QVector<Bone*> boneList;
	skeleton->makeBoneListFrom(baseBone, boneList);
	Transforms.resize(skeleton->getSkeletonSize());
	for (int i = 0; i < boneList.size(); ++i)
	{
		Transforms[boneList[i]->m_ID] = boneList[i]->m_finalTransform;
	}
}
