#pragma once
#include "BoneInfo.h"
#include <Utility/DataTypes.h>

class Skeleton
{
public:
	Skeleton(Bone* root);
	~Skeleton();

	Bone* getBone(QString boneName);

	// calculate and set the global transformation
	// for each bone in the skeleton
	Bone* sortSkeleton(Bone* root);
	mat4 calcGlobalTransformation(Bone* bone);


	// clean up the skeleton
	Bone* freeSkeleton(Bone* root);
	

	/** Method to print out the skeleton. **/
	void dumpSkeleton(Bone* pBone, uint level);

	bool isBoneInSkeleton(const QString& boneName);

	bool isInTheSameChain(Bone* upperBone, Bone* lowerBone);
	bool isInTheSameChain(const QString& upperBoneName, const QString& lowerBoneName);

	float getDistanceBetween(Bone* upperBone, Bone* lowerBone);
	float getDistanceBetween(const QString& upperBoneName, const QString& lowerBoneName);

	uint getBoneCountBetween(Bone* upperBone, Bone* lowerBone);

private:
	

	/** The root bone of the skeleton. **/
	Bone* m_root;

	/** The global inverse matrix, due to Assimp. **/
	mat4 m_gloableInverseMatrix;

	/** The bone map witch stores the bone name and its pointer. **/
	QMap<QString, Bone*> m_BoneMap;
};

