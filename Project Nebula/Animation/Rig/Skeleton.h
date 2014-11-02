#pragma once
#include "BoneInfo.h"
#include <Utility/DataTypes.h>

class Skeleton
{
public:
	Skeleton();
	Skeleton(BoneInfo &bi, Skeleton* parent);
	~Skeleton();

	void addChild(Skeleton* child);
	Skeleton* getChild(uint i);
	QVector<Skeleton*> getChildren();
	int childCount();

	// clean up the skeleton
	static Skeleton* freeSkeleton(Skeleton* root);

private:
	/** The bone info of the skeleton. **/
	BoneInfo m_boneInfo;

	/** Parent bode. NULL if this node is the root. **/
	Skeleton* m_parent;

	/** The child bones of this bone. **/
	QVector<Skeleton*> m_children;
	
};

