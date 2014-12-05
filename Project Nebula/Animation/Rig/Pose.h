#pragma once
#include <Animation/Rig/Skeleton.h>

class Pose
{
public:
	Pose();
	Pose(QVector<Bone*>& boneList);
	~Pose();
	QVector<Bone*> getBoneList() { return m_boneList; }
	void setBoneList(QVector<Bone*>& boneList){ m_boneList = boneList; }

	// static function
	static Pose lerp(Pose from, Pose to, float fraction);

private:
	QVector<Bone*> m_boneList;
};

