#pragma once
#include "BoneInfo.h"
#include <Utility/EngineCommon.h>

class Skeleton
{
public:
	Skeleton(Bone* root, mat4& globalInverseMatrix);
	~Skeleton();

	Bone* getBone(QString boneName);
	int getSkeletonSize();
	QMap<QString, Bone*> getBoneMap();
	QVector<Bone*> getBoneList();
	void makeBoneListFrom(Bone* baseBone, QVector<Bone*> &listOut);
	
	Bone* sortSkeleton(Bone* root);
	void sortPose(Bone* pBone, mat4 &parentTransform);


	// clean up the skeleton
	Bone* freeSkeleton(Bone* root);
	
	mat4 getBoneGlobalTransform(Bone* pBone);

	/** Method to print out the skeleton. **/
	void dumpSkeleton(Bone* pBone, uint level);

	bool isBoneInSkeleton(const QString& boneName);

	bool isInTheSameChain(Bone* upperBone, Bone* lowerBone);
	bool isInTheSameChain(const QString& upperBoneName, const QString& lowerBoneName);

	float getDistanceBetween(Bone* upperBone, Bone* lowerBone);
	float getDistanceBetween(const QString& upperBoneName, const QString& lowerBoneName);

	uint getBoneCountBetween(Bone* upperBone, Bone* lowerBone);

	Bone* getRoot() { return m_root; }

	void getBoneChain(Bone* start, Bone* end, QVector<Bone*> &boneChain);

	const mat4 getGlobalInverseMatrix() { return m_gloableInverseMatrix; }

private:

	void initialize(Bone* pBone);

	/** The root bone of the skeleton. **/
	Bone* m_root;

	/** The global inverse matrix, due to Assimp. **/
	mat4 m_gloableInverseMatrix;

	/** The bone map witch stores the bone name and its pointer. **/
	QMap<QString, Bone*> m_BoneMap;
	QVector<Bone*> m_BoneList;
};

