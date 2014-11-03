#include "Skeleton.h"
#include <stdio.h>


Skeleton::Skeleton( Bone* root )
{
	m_gloableInverseMatrix = mat4(1, 0, 0, 0, 
								  0, 0, -1, 0,
								  0, 1, 0, 0,
								  0, 0, 0, 1);
	m_root = root;
	m_root->m_finalTransform = m_gloableInverseMatrix * m_root->m_finalTransform;
	sortSkeleton(m_root);
}


Skeleton::~Skeleton()
{
	freeSkeleton(m_root);
}

void Skeleton::dumpSkeleton( Bone* pBone, uint level )
{
	for (uint i = 0; i < level; ++i)
		printf("  ");
	
	qDebug() << pBone->m_ID << pBone->m_name;

	for (int i = 0; i < pBone->childCount(); ++i)
		dumpSkeleton(pBone->getChild(i), level + 1);
}


Bone* Skeleton::freeSkeleton( Bone* root )
{
	if(!root) return NULL; // empty skeleton
	free(root);
	for (int i = 0; i < root->childCount(); ++i)
	{
		freeSkeleton(root->getChild(i));
	}

	return NULL;
}

Bone* Skeleton::sortSkeleton( Bone* root )
{
	if(!root) return NULL; // empty skeleton

	// calculate the final transform
	root->m_finalTransform = calcGlobalTransformation(root);

	// calculate the world position
	root->calcWorldPos(); // need to fix

	// store this bone into the map
	if (m_BoneMap.find(root->m_name) == m_BoneMap.end())
		m_BoneMap[root->m_name] = root;

	for (int i = 0; i < root->childCount(); ++i)
	{
		sortSkeleton(root->getChild(i));
	}
	return NULL;
}

mat4 Skeleton::calcGlobalTransformation( Bone* bone )
{
	// empty bone
	mat4 result;
	if (!bone) return result;

	result = bone->getLocalTransformMatrix();
	while (bone->m_parent)
	{
		// NOTE: the matrix multiplying order is very important!
		result = bone->m_parent->getLocalTransformMatrix() * result;
		bone = bone->m_parent;
	}
	return result;
}

float Skeleton::getDistanceBetween( Bone* upperBone, Bone* lowerBone )
{
	// if they are the same bone
	if (upperBone == lowerBone) return 0.0f;

	// check if they are in the same chain
	if (!isInTheSameChain(upperBone, lowerBone))
	{
		qDebug() << "getLengthBetween Failed! The Bones are not in the same  Chain" << upperBone->m_name << lowerBone->m_name;
		return -1;
	}

	// calculate the world distance
	return (upperBone->getWorldPosition() - lowerBone->getWorldPosition()).length();
}

float Skeleton::getDistanceBetween( const QString& upperBoneName, const QString& lowerBoneName )
{
	// check input
	if(!isBoneInSkeleton(lowerBoneName) || !isBoneInSkeleton(upperBoneName)) return -1.0;

	Bone* upperBone = m_BoneMap[upperBoneName];
	Bone* lowerBone = m_BoneMap[lowerBoneName];

	return getDistanceBetween(upperBone, lowerBone);
}

Bone* Skeleton::getBone( QString boneName )
{
	if (isBoneInSkeleton(boneName)) return m_BoneMap[boneName];
	else return NULL;
}


bool Skeleton::isBoneInSkeleton( const QString& boneName )
{
	if (m_BoneMap.find(boneName) == m_BoneMap.end())
	{
		qDebug() << "Cannot find bone" << boneName << "in the skeleton!";
		return false;
	}
	return true;
}

bool Skeleton::isInTheSameChain( Bone* upperBone, Bone* lowerBone )
{
	// same bone
	if(upperBone == lowerBone) return false;
	
	bool inTheSameChain = false;
	Bone* temp = lowerBone;
	while(temp->m_parent)
	{
		if (temp->m_parent == upperBone)
		{
			inTheSameChain = true;
			break;
		}

		temp = temp->m_parent;
	}

	return inTheSameChain;
}

bool Skeleton::isInTheSameChain( const QString& upperBoneName, const QString& lowerBoneName )
{
	// check input
	if(!isBoneInSkeleton(lowerBoneName) || !isBoneInSkeleton(upperBoneName)) return false;

	return isInTheSameChain(m_BoneMap[upperBoneName], m_BoneMap[lowerBoneName]);
}

uint Skeleton::getBoneCountBetween( Bone* upperBone, Bone* lowerBone )
{
	if (upperBone == lowerBone || !isInTheSameChain(upperBone, lowerBone))
	{
		return 0;
	}

	uint count = 1;
	Bone* temp = lowerBone;
	while(temp != upperBone)
	{
		++count;
		temp = temp->m_parent;
	}

	return count;
}
