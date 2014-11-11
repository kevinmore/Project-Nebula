#include "CCDIKSolver.h"


CCDIKSolver::CCDIKSolver( int iterations )
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

// 	QQuaternion test = Math::QuaternionFromEuler(Math::EulerAngle(0, 0,1));
// 	skeleton->getBone("Bip01_L_Finger2")->rotateInWorldSpace(test);
// 	return true;

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
		return false;
	}
	
	// begin the iteration
	for( int iteration = 0; iteration < m_iterations; ++iteration )
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
			float cosAngle = vec3::dotProduct(currentDirection,  targetDirenction);

			// 360 degree
			if (cosAngle >= 0.999f) continue;
			float deltaAngle = qRadiansToDegrees(qAcos(cosAngle));
			// if the angle is too small, or greater than 180 degree(that's not real for a human)
			if (deltaAngle < 1.0f || qAbs(deltaAngle) > 179.99f || Math::isNaN(deltaAngle))
			{  
				continue;
			}

			// create the delta quaternion
			QQuaternion deltaRotation = QQuaternion::fromAxisAndAngle(rotationAxis, deltaAngle);


			Bone::DimensionOfFreedom dof = joint->getDof();
			QQuaternion curRotation;
			Math::EulerAngle eulerAngles;
			float curYaw, curPitch, curRoll;  
			float deltaYaw, deltaPitch, deltaRoll;

			// Check DOF
			// get the current euler angles
			eulerAngles = joint->getLocalEulerAngleInDegrees();
			curRoll  = eulerAngles.m_fRoll;
			curPitch = eulerAngles.m_fPitch;
			curYaw   = eulerAngles.m_fYaw;

			

// 			// decompose the delta rotation
// 			eulerAngles = Math::QuaternionToEuler(deltaRotation);
// 			deltaRoll  = qRadiansToDegrees(eulerAngles.m_fRoll);
// 			deltaPitch = qRadiansToDegrees(eulerAngles.m_fPitch);
// 			deltaYaw   = qRadiansToDegrees(eulerAngles.m_fYaw);
// 
// 			// bound the delta rotation
// 			// Min <= curAngle + deltaAngle <= Max
// 			// which is, Min - curAngle <= deltaAngle <= Max - curAngle
// 			deltaRoll  = qBound(dof.RollConstraint.minAngle - curRoll, deltaRoll, dof.RollConstraint.maxAngle - curRoll);
// 			deltaPitch = qBound(dof.PitchConstraint.minAngle - curPitch, deltaPitch, dof.PitchConstraint.maxAngle - curPitch);
// 			deltaYaw   = qBound(dof.YawConstraint.minAngle - deltaYaw, deltaRoll, dof.YawConstraint.maxAngle - deltaYaw);
// 
// 			// remake the quaternion
// 			deltaRotation = Math::QuaternionFromEuler(Math::EulerAngle(deltaRoll, deltaPitch, deltaYaw));

			
			// adjust the world rotation of the joint
			joint->rotateInWorldSpace(deltaRotation);
			// re-sort the skeleton pose
			skeleton->sortPose(baseBone, baseBone->m_parent->m_globalNodeTransform);
		}
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
