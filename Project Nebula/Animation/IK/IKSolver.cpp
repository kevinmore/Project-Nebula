#include "IKSolver.h"


IKSolver::IKSolver(Skeleton* skeleton)
	: m_skeleton(skeleton)
{}


IKSolver::~IKSolver(void)
{
}

void IKSolver::solveIK( const QString &effectorName, const QString &rootName, const vec3 &targetPos )
{

	/************************************************************************/
	/* Check inputs                                                         */
	/************************************************************************/
	Bone* rootBone = m_skeleton->getBone(rootName);
	Bone* effectorBone = m_skeleton->getBone(effectorName);
	if(!effectorBone || !rootBone)
	{
		qDebug() << "solveIK failed! Can't find bones.";
		return;
	}	

	// check if the two bones are in the same chain
	if(!m_skeleton->isInTheSameChain(rootName, effectorName))
	{
		qDebug() << "solveIK failed! Bone" << rootName << "and Bone" << effectorName << "are not in the same chain." ;
		return;
	}


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
	/* Calculate the original distances between each bone                   */
	/************************************************************************/
	// from root to effector
	QVector<float> distances;
	for(int i = 0; i < m_boneChain.size() - 1; ++i)
		distances.push_back(m_skeleton->getDistanceBetween(m_boneChain[i], m_boneChain[i + 1]));


	/************************************************************************/
	/* Check if the target is reachable                                     */
	/************************************************************************/
	vec3 originalRootPosition = rootBone->getWorldPosition();
	float totalChainLength = 0.0f;
	for (int i = 0; i < distances.size(); ++i)
	{
		totalChainLength += distances[i];
	}

	float rootToTargetLenght = (targetPos - originalRootPosition).length();
	if(totalChainLength - rootToTargetLenght < 0.00001f)
	{
		qDebug() << "solveIK failed! Target out of range.";
		return;
	}


	/************************************************************************/
	/* Stage 1: Forward Reaching                                            */
	/************************************************************************/
	uint effectorIndex = chainCount - 1;
	effectorBone->setWorldPosition(targetPos);
	vec3 direction;
	Bone *currentBone, *parentToCurrent;
	uint i = effectorIndex;
	bool test = i > effectorIndex - chainCount;

	// from effector to root
	for (uint i = effectorIndex; i > 0; --i)
	{
		currentBone = m_boneChain[i];
		parentToCurrent = m_boneChain[i-1];

		direction = (parentToCurrent->getWorldPosition() - currentBone->getWorldPosition()).normalized();
		parentToCurrent->setWorldPosition(currentBone->getWorldPosition() + direction * distances[i-1]);
	}

	/************************************************************************/
	/* Stage 2: Backward Reaching                                           */
	/************************************************************************/
	rootBone->setWorldPosition(originalRootPosition);
	Bone* childOfCurrent;

	// from root to effector
	for (uint i = 0; i < effectorIndex; ++i)
	{
		currentBone = m_boneChain[i];
		childOfCurrent = m_boneChain[i+1];

		direction = (childOfCurrent->getWorldPosition() - currentBone->getWorldPosition()).normalized();
		childOfCurrent->setWorldPosition(currentBone->getWorldPosition() + direction * distances[i]);
	}

	/************************************************************************/
	/* Stage 3: Moving the children of the effector(if any)                 */
	/************************************************************************/
	for (int i = 0; i < effectorBone->childCount(); ++i)
	{
		m_skeleton->calcGlobalTransformation(effectorBone->getChild(i));
	}

}
