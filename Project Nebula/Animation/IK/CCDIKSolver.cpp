#include "CCDIKSolver.h"


CCDIKSolver::CCDIKSolver( int iterations )
	: m_iterations( iterations ),
	  m_lastDistance( 0.0f )
{}

CCDIKSolver::~CCDIKSolver()
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
	skeleton->sortPose(baseBone, baseBone->m_parent->m_modelSpaceTransform);

	// find the set of bones within this chain
	QVector<Bone*> boneChain;
	skeleton->getBoneChain(baseBone, effectorBone, boneChain);

	// if there are bones in the chain
	if (!boneChain.size()) return false;

	QVector<float> bone_distances;
	float totalChainLength;
	bone_distances.clear();
	for(int i = 0; i < boneChain.size() - 1; ++i)
		bone_distances.push_back(skeleton->getDistanceBetween(boneChain[i], boneChain[i + 1]));

	totalChainLength = 0.0f;
	for (int i = 0; i < bone_distances.size(); ++i)
	{
		totalChainLength += bone_distances[i];
	}

	float rootToTargetLenght = (constraint.m_targetMS - baseBone->getModelSpacePosition()).length();

	if(totalChainLength < rootToTargetLenght + 10.0f) // to avoid funny results
		return false;

	// check if the last distance is about the same
	float effectorToTarget = (constraint.m_targetMS - effectorBone->getModelSpacePosition()).length();
	if(qAbs(m_lastDistance - effectorToTarget) < 0.5f)
		return true;

	// begin the iteration
	for( int iteration = 0; iteration < m_iterations; ++iteration )
	{
		// skip the effector
		for( int jointIndex = boneChain.size() - 2; jointIndex >= 0; --jointIndex )
		{
			// check if the target is already reached
			if ((effectorBone->getModelSpacePosition() - constraint.m_targetMS).length() < 1.0f)
				return true;
			
			// the joint to rotate
			Bone* joint = boneChain[jointIndex];


			const vec3 effectorPos = effectorBone->getModelSpacePosition();
			const vec3 jointPos = joint->getModelSpacePosition();

			// joint to effector bone direction
			vec3 currentDirection = (effectorPos - jointPos).normalized();

			// joint to target direction
			vec3 targetDirenction = (constraint.m_targetMS - jointPos).normalized();

			// calculate the rotation axis and angle
			const vec3 rotationAxis = vec3::crossProduct(currentDirection, targetDirenction);
			float cosAngle = vec3::dotProduct(currentDirection,  targetDirenction);

			// 360 degree
			if (cosAngle >= 0.99f) continue;
			float deltaAngle = qRadiansToDegrees(qAcos(cosAngle));
			// if the angle is too small, or greater than 180 degree(that's not real for a human)
			if (deltaAngle < 0.1f ||  Math::isNaN(deltaAngle))
			{  
				continue;
			}

			// create the delta quaternion
			QQuaternion deltaRotation = QQuaternion::fromAxisAndAngle(rotationAxis, deltaAngle);


// 			Bone::DimensionOfFreedom dof = joint->getDof();
// 			QQuaternion curRotation;
// 			Math::EulerAngle eulerAngles;
// 			float curYaw, curPitch, curRoll;  
// 			float deltaYaw, deltaPitch, deltaRoll;
// 
// 			// Check DOF
// 			// get the current euler angles
// 			eulerAngles = joint->getGlobalAngleInDegrees();
// 			curRoll  = eulerAngles.m_fRoll;
// 			curPitch = eulerAngles.m_fPitch;
// 			curYaw   = eulerAngles.m_fYaw;
// 
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
			joint->rotateInModelSpace(deltaRotation);
			// re-sort the skeleton pose
			skeleton->sortPose(baseBone, baseBone->m_parent->m_modelSpaceTransform);
			
		}// end of 1 iteration
	}// end of iterations

	m_lastDistance = (constraint.m_targetMS - effectorBone->getModelSpacePosition()).length();

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
