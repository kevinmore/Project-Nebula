#include "Skeleton.h"
#include <stdio.h>


Skeleton::Skeleton( Bone* root , mat4& globalInverseMatrix)
	: m_root(root),
	  m_gloableInverseMatrix(globalInverseMatrix)
{
	mat4 identity;
	sortPose(m_root, identity);

	// generate the bone list
	// the root skeleton is added manually in the loader
	// that's why here we minus 1
	m_BoneList.resize(getSkeletonSize() - 1);
	QMap<QString, Bone*>::iterator it;
	for (it = m_BoneMap.begin(); it != m_BoneMap.end(); ++it)
	{
		if(it.value()->m_name == "Project Nebula Skeleton ROOT") continue;
		m_BoneList[it.value()->m_ID] = it.value();
	}

// 	for (int i = 0; i < m_BoneList.size(); ++i)
// 	{

// 			qDebug() << "offset:"<<m_BoneList[i]->m_offsetMatrix;
// 			qDebug() << "node:"<<m_BoneList[i]->m_nodeTransform;
// 			qDebug() << "node*offset"<<m_BoneList[i]->m_nodeTransform * m_BoneList[i]->m_offsetMatrix;
// 			qDebug() <<m_BoneList[i]->m_name;
// 			qDebug() << "final:"<<m_BoneList[i]->m_finalTransform;

			
		
			
		//qDebug() << m_BoneList[i]->m_ID << m_BoneList[i]->m_name << m_BoneList[i]->getWorldPosition();
		//qDebug() << m_BoneList[i]->m_finalTransform;
//	}

}


Skeleton::~Skeleton()
{
	freeSkeleton(m_root);
}

void Skeleton::sortPose( Bone* pBone, mat4 &parentTransform )
{
	// add this bone to the map
	if (m_BoneMap.find(pBone->m_name) == m_BoneMap.end())
		m_BoneMap[pBone->m_name] = pBone;

	// calculate the global transform
	mat4 globalTransformation = parentTransform * pBone->m_nodeTransform;  // P * B
	pBone->m_globalNodeTransform = globalTransformation;
	pBone->m_finalTransform = m_gloableInverseMatrix* globalTransformation * pBone->m_offsetMatrix;

	pBone->calcWorldTransform();

	for (int i = 0 ; i < pBone->childCount() ; ++i) 
	{
		sortPose(pBone->getChild(i), globalTransformation);
	}
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

	return getDistanceBetween(m_BoneMap[upperBoneName], m_BoneMap[lowerBoneName]);
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

int Skeleton::getSkeletonSize()
{
	return m_BoneMap.size();
}

QMap<QString, Bone*> Skeleton::getBoneMap()
{
	return m_BoneMap;
}

QVector<Bone*> Skeleton::getBoneList()
{
	return m_BoneList;
}

mat4 Skeleton::getBoneGlobalTransform( Bone* pBone )
{
	mat4 globalTransform;

	while(pBone->m_parent && pBone->m_parent->m_name != "Project Nebula Skeleton ROOT")
	{
		globalTransform = pBone->m_parent->m_offsetMatrix * globalTransform;
		pBone = pBone->m_parent;
	}

	return globalTransform;
}

void Skeleton::applyOffset( Bone* pBone, mat4& offset )
{
	if(!pBone) return; // empty skeleton
	pBone->m_offsetMatrix *= offset;

	for (int i = 0; i < pBone->childCount(); ++i)
	{
		applyOffset(pBone->getChild(i), offset);
	}
}

void Skeleton::getBoneChain( Bone* start, Bone* end, QVector<Bone*> &boneChain )
{
	uint chainCount = getBoneCountBetween(start, end);
	boneChain.resize(chainCount);
	uint index = chainCount - 1;
	Bone* temp = end;
	while(temp != start->m_parent)
	{
		boneChain[index] = temp;
		--index;
		temp = temp->m_parent;
	}
}


void Skeleton::makeBoneListFrom( Bone* baseBone, QVector<Bone*> &listOut )
{
	if (!baseBone) return;

	listOut.push_back(baseBone);

	for (int i = 0; i < baseBone->childCount(); ++i)
	{
		makeBoneListFrom(baseBone->getChild(i), listOut);
	}

}


