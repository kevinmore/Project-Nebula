#include "FABRIKSolver.h"


FABRIKSolver::FABRIKSolver(Skeleton* skeleton, float tolerance)
	: m_skeleton(skeleton),
	  m_tolerance(tolerance)
{}


FABRIKSolver::~FABRIKSolver(void)
{
}

bool FABRIKSolver::enableIKChain( const QString &rootName, const QString &effectorName )
{
	/************************************************************************/
	/* Check inputs                                                         */
	/************************************************************************/
	Bone* rootBone = m_skeleton->getBone(rootName);
	Bone* effectorBone = m_skeleton->getBone(effectorName);
	if(!effectorBone || !rootBone)
	{
		qDebug() << "solveIK failed! Can't find bones.";
		return false;
	}	

	// check if the two bones are in the same chain
	if(!m_skeleton->isInTheSameChain(rootName, effectorName))
	{
		qDebug() << "solveIK failed! Bone" << rootName << "and Bone" << effectorName << "are not in the same chain." ;
		return false;
	}

	m_rootBone = rootBone;
	m_effectorBone = effectorBone;
	/************************************************************************/
	/* Create the bone chain                                                */
	/************************************************************************/
	uint chainCount = m_skeleton->getBoneCountBetween(rootBone, effectorBone);
	m_boneChain.resize(chainCount);
	uint index = chainCount - 1;
	Bone* temp = effectorBone;
	while(temp != rootBone->m_parent)
	{
		m_boneChain[index] = temp;
		--index;
		temp = temp->m_parent;
	}


	/************************************************************************/
	/* Calculate the original distances, and the total chain length         */
	/************************************************************************/
	// from root to effector
	m_distances.clear();
	for(int i = 0; i < m_boneChain.size() - 1; ++i)
		m_distances.push_back(m_skeleton->getDistanceBetween(m_boneChain[i], m_boneChain[i + 1]));

	m_totalChainLength = 0.0f;
	for (int i = 0; i < m_distances.size(); ++i)
	{
		m_totalChainLength += m_distances[i];
	}
	
	return true;
}

void FABRIKSolver::solveIK( const vec3 &targetPos )
{

	/************************************************************************/
	/* Check if the target is already reached                               */
	/************************************************************************/
	if((m_effectorBone->getModelSpacePosition() - targetPos).length() <= m_tolerance)
	{
		qDebug() << "Target Reached";
		return;
	}

	/************************************************************************/
	/* Check if the target is reachable                                     */
	/************************************************************************/
	vec3 originalRootPosition = m_rootBone->getModelSpacePosition();
	
	float rootToTargetLenght = (targetPos - originalRootPosition).length();
	if(m_totalChainLength - rootToTargetLenght < 0.00001f)
	{
		//qDebug() << "solveIK failed! Target out of range.";
		return;
	}

// 	qDebug() << "Original Bone Chain.";
// 	for (int i = 0; i < m_boneChain.size(); ++i)
// 	{
// 		qDebug() << "Bone[" << i << "]" << m_boneChain[i]->getWorldPosition();
// 	}

	/************************************************************************/
	/* Stage 1: Forward Reaching                                            */
	/************************************************************************/
	uint chainCount = m_skeleton->getBoneCountBetween(m_rootBone, m_effectorBone);
	uint effectorIndex = chainCount - 1;
	m_effectorBone->setModelSpacePosition(targetPos);
	vec3 direction;
	Bone *currentBone, *parentToCurrent;
	uint i = effectorIndex;
	bool test = i > effectorIndex - chainCount;

	// from effector to root
	for (uint i = effectorIndex; i > 0; --i)
	{
		currentBone = m_boneChain[i];
		parentToCurrent = m_boneChain[i-1];

		direction = (parentToCurrent->getModelSpacePosition() - currentBone->getModelSpacePosition()).normalized();
		parentToCurrent->setModelSpacePosition(currentBone->getModelSpacePosition() + direction * m_distances[i-1]);
	}


// 	qDebug() << "After Stage 1 Bone Chain.";
// 	for (int i = 0; i < m_boneChain.size(); ++i)
// 	{
// 		qDebug() << "Bone[" << i << "]" << m_boneChain[i]->getWorldPosition();
// 	}
	/************************************************************************/
	/* Stage 2: Backward Reaching                                           */
	/************************************************************************/
	m_rootBone->setModelSpacePosition(originalRootPosition);
	Bone* childOfCurrent;

	// from root to effector
	for (uint i = 0; i < effectorIndex; ++i)
	{
		currentBone = m_boneChain[i];
		childOfCurrent = m_boneChain[i+1];

		direction = (childOfCurrent->getModelSpacePosition() - currentBone->getModelSpacePosition()).normalized();
		childOfCurrent->setModelSpacePosition(currentBone->getModelSpacePosition() + direction * m_distances[i]);
	}


// 	qDebug() << "After Stage 2 Bone Chain.";
// 	for (int i = 0; i < m_boneChain.size(); ++i)
// 	{
// 		qDebug() << "Bone[" << i << "]" << m_boneChain[i]->getWorldPosition();
// 	}
// 	printf("\n");

	

	/************************************************************************/
	/* Stage 3: Update the skeleton position                                */
	/************************************************************************/
	m_skeleton->sortPose(m_rootBone, m_rootBone->m_parent->m_modelSpaceTransform);
}

void FABRIKSolver::BoneTransform( Skeleton* skeleton, Bone* baseBone, QVector<mat4>& Transforms )
{

	QVector<Bone*> boneList;
	skeleton->makeBoneListFrom(baseBone, boneList);
	Transforms.resize(skeleton->getSkeletonSize());
	for (int i = 0; i < boneList.size(); ++i)
	{
		Transforms[boneList[i]->m_ID] = boneList[i]->m_finalTransform;
	}
}
