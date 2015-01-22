#pragma once
#include <Utility/ModelLoader.h>
#include <assimp/scene.h>

class FKController
{
public:
	FKController(ModelLoaderPtr loader, Skeleton* skeleton);
	~FKController();

	void getBoneTransforms(float TimeInSeconds, QVector<mat4>& Transforms);
	void enableAllBones();
	void disableBoneChain(Bone* baseBone);
	inline vec3 getRootTranslation() { return m_rootPositionTranslation; }
	inline QQuaternion getRootRotation() { return m_rootRotationTranslation; }

private:
	void interpolateScaling(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
	void interpolateRotation(aiQuaternion& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
	void interpolatePosition(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);    
	uint findScaling(float AnimationTime, const aiNodeAnim* pNodeAnim);
	uint findRotation(float AnimationTime, const aiNodeAnim* pNodeAnim);
	uint findPosition(float AnimationTime, const aiNodeAnim* pNodeAnim);
	const aiNodeAnim* findNodeAnim(const aiAnimation* pAnimation, const QString NodeName);
	void calcFinalTransforms(float AnimationTime, const aiNode* pNode, const mat4& ParentTransform);

	QMap<QString, uint> m_BoneMapping; // maps a bone name to its index
	mat4 m_GlobalInverseTransform;
	uint m_NumBones;
	QVector<Bone*> m_BoneInfo;
	aiAnimation** m_Animations;
	aiNode* m_root;
	Skeleton* m_skeleton;

	QVector<Bone*> m_disabledBones;
	vec3 m_rootPositionTranslation;
	QQuaternion m_rootRotationTranslation;
};

