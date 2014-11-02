#include "Skeleton.h"


Skeleton::Skeleton()
	: m_parent(NULL)
{}

Skeleton::Skeleton( BoneInfo &bi, Skeleton* parent )
	: m_parent(parent),
	  m_boneInfo(bi)
{
	// add the current bone to its parent if it's not the root
	if(parent) parent->addChild(this);
}


Skeleton::~Skeleton(void)
{
	freeSkeleton(this);
}

void Skeleton::addChild( Skeleton* child )
{
	m_children.push_back(child);
}

QVector<Skeleton*> Skeleton::getChildren()
{
	return m_children;
}

Skeleton* Skeleton::getChild( uint i )
{
	if(m_children.isEmpty()) return NULL;
	else return m_children.at(i);
}

int Skeleton::childCount()
{
	return m_children.size();
}

Skeleton* Skeleton::freeSkeleton( Skeleton* root )
{
	if(!root) return NULL; // empty skeleton
	free(root);
	for (int i = 0; i < root->childCount(); ++i)
	{
		freeSkeleton(root->getChild(i));
	}

	return NULL;
}
