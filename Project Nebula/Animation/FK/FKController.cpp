#include "FKController.h"
#include <Utility/Math.h>

FKController::FKController(ModelLoader* loader, Skeleton* skeleton)
{
	m_GlobalInverseTransform = loader->getGlobalInverseTransform();
	m_BoneMapping = loader->getBoneMap();
	m_NumBones = loader->getNumBones();
	m_Animations = loader->getAnimations();
	m_root = loader->getRootNode();
	m_skeleton = skeleton;
	m_BoneInfo = skeleton->getBoneList();
}

FKController::~FKController(void)
{}

void FKController::getBoneTransforms( float TimeInSeconds, QVector<mat4>& Transforms )
{

	mat4 Identity;
	Identity.setToIdentity();
	float TicksPerSecond = (float)(m_Animations[0]->mTicksPerSecond != 0 ? m_Animations[0]->mTicksPerSecond : 25.0f);
	float TimeInTicks = TimeInSeconds * TicksPerSecond;
	float AnimationTime = fmod(TimeInTicks, (float)m_Animations[0]->mDuration);
	calcFinalTransforms(AnimationTime, m_root, Identity);

	Transforms.resize(m_NumBones);

	for (uint i = 0 ; i < m_NumBones ; i++)
	{
		// only update the bones that are enabled
		// that is, the bone is not in the disabled bones list
		if (m_disabledBones.indexOf(m_BoneInfo[i]) != -1) continue;
		Transforms[i] = m_BoneInfo[i]->m_finalTransform;
	}
}

void FKController::calcFinalTransforms( float AnimationTime, const aiNode* pNode, const mat4 &ParentTransform )
{
	QString NodeName(pNode->mName.data);

	const aiAnimation* pAnimation = m_Animations[0];

	mat4 NodeTransformation(Math::convToQMat4(pNode->mTransformation));
	
	const aiNodeAnim* pNodeAnim = findNodeAnim(pAnimation, NodeName);
	if (pNodeAnim) {
		// Interpolate scaling and generate scaling transformation matrix
		aiVector3D Scaling;
		interpolateScaling(Scaling, AnimationTime, pNodeAnim);
		mat4 ScalingM;
		ScalingM.scale(Scaling.x, Scaling.y, Scaling.z);

		// Interpolate rotation and generate rotation transformation matrix
		aiQuaternion RotationQ;
		interpolateRotation(RotationQ, AnimationTime, pNodeAnim);        
		mat4 RotationM = Math::convToQMat4(RotationQ.GetMatrix());

		// Interpolate translation and generate translation transformation matrix
		aiVector3D Translation;
		interpolatePosition(Translation, AnimationTime, pNodeAnim);
		mat4 TranslationM;
		TranslationM.translate(Translation.x, Translation.y, Translation.z);

		// Combine the above transformations
		NodeTransformation = TranslationM * RotationM * ScalingM;

	}


	 mat4 GlobalTransformation = ParentTransform * NodeTransformation;
	 if (m_BoneMapping.find(NodeName) != m_BoneMapping.end()) 
	 {
		uint BoneIndex = m_BoneMapping[NodeName];

		// only update the bones that are enabled
		// that is, the bone is not in the disabled bones list
		if (m_disabledBones.indexOf(m_BoneInfo[BoneIndex]) == -1)
		{
			m_BoneInfo[BoneIndex]->m_nodeTransform = NodeTransformation;
			m_BoneInfo[BoneIndex]->m_globalNodeTransform = GlobalTransformation;
			m_BoneInfo[BoneIndex]->m_finalTransform = m_GlobalInverseTransform * GlobalTransformation * m_BoneInfo[BoneIndex]->m_offsetMatrix;

			// update the global position
			m_BoneInfo[BoneIndex]->calcWorldTransform();
		}

	 }

	for (uint i = 0 ; i < pNode->mNumChildren ; i++) 
	{
		calcFinalTransforms(AnimationTime, pNode->mChildren[i], GlobalTransformation);
	}

}

void FKController::interpolatePosition(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim)
{
	if (pNodeAnim->mNumPositionKeys == 1) {
		Out = pNodeAnim->mPositionKeys[0].mValue;
		return;
	}

	uint PositionIndex = findPosition(AnimationTime, pNodeAnim);
	uint NextPositionIndex = (PositionIndex + 1);
	assert(NextPositionIndex < pNodeAnim->mNumPositionKeys);
	float DeltaTime = (float)(pNodeAnim->mPositionKeys[NextPositionIndex].mTime - pNodeAnim->mPositionKeys[PositionIndex].mTime);
	float Factor = (AnimationTime - (float)pNodeAnim->mPositionKeys[PositionIndex].mTime) / DeltaTime;
	//assert(Factor >= 0.0f && Factor <= 1.0f);
	const aiVector3D& Start = pNodeAnim->mPositionKeys[PositionIndex].mValue;
	const aiVector3D& End = pNodeAnim->mPositionKeys[NextPositionIndex].mValue;
	aiVector3D Delta = End - Start;
	Out = Start + Factor * Delta;
}

void FKController::interpolateRotation(aiQuaternion& Out, float AnimationTime, const aiNodeAnim* pNodeAnim)
{
	// we need at least two values to interpolate...
	if (pNodeAnim->mNumRotationKeys == 1) {
		Out = pNodeAnim->mRotationKeys[0].mValue;
		return;
	}

	uint RotationIndex = findRotation(AnimationTime, pNodeAnim);
	uint NextRotationIndex = (RotationIndex + 1);
	assert(NextRotationIndex < pNodeAnim->mNumRotationKeys);
	float DeltaTime = (float)(pNodeAnim->mRotationKeys[NextRotationIndex].mTime - pNodeAnim->mRotationKeys[RotationIndex].mTime);
	float Factor = (AnimationTime - (float)pNodeAnim->mRotationKeys[RotationIndex].mTime) / DeltaTime;
	//assert(Factor >= 0.0f && Factor <= 1.0f);
	const aiQuaternion& StartRotationQ = pNodeAnim->mRotationKeys[RotationIndex].mValue;
	const aiQuaternion& EndRotationQ   = pNodeAnim->mRotationKeys[NextRotationIndex].mValue;    
	aiQuaternion::Interpolate(Out, StartRotationQ, EndRotationQ, Factor);
	Out = Out.Normalize();
}

void FKController::interpolateScaling(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim)
{
	if (pNodeAnim->mNumScalingKeys == 1) {
		Out = pNodeAnim->mScalingKeys[0].mValue;
		return;
	}

	uint ScalingIndex = findScaling(AnimationTime, pNodeAnim);
	uint NextScalingIndex = (ScalingIndex + 1);
	assert(NextScalingIndex < pNodeAnim->mNumScalingKeys);
	float DeltaTime = (float)(pNodeAnim->mScalingKeys[NextScalingIndex].mTime - pNodeAnim->mScalingKeys[ScalingIndex].mTime);
	float Factor = (AnimationTime - (float)pNodeAnim->mScalingKeys[ScalingIndex].mTime) / DeltaTime;
	//assert(Factor >= 0.0f && Factor <= 1.0f);
	const aiVector3D& Start = pNodeAnim->mScalingKeys[ScalingIndex].mValue;
	const aiVector3D& End   = pNodeAnim->mScalingKeys[NextScalingIndex].mValue;
	aiVector3D Delta = End - Start;
	Out = Start + Factor * Delta;
}

uint FKController::findPosition(float AnimationTime, const aiNodeAnim* pNodeAnim)
{    
	for (uint i = 0 ; i < pNodeAnim->mNumPositionKeys - 1 ; i++) {
		if (AnimationTime < (float)pNodeAnim->mPositionKeys[i + 1].mTime) {
			return i;
		}
	}

	assert(0);

	return 0;
}

uint FKController::findRotation(float AnimationTime, const aiNodeAnim* pNodeAnim)
{
	assert(pNodeAnim->mNumRotationKeys > 0);

	for (uint i = 0 ; i < pNodeAnim->mNumRotationKeys - 1 ; i++) {
		if (AnimationTime < (float)pNodeAnim->mRotationKeys[i + 1].mTime) {
			return i;
		}
	}

	assert(0);

	return 0;
}

uint FKController::findScaling(float AnimationTime, const aiNodeAnim* pNodeAnim)
{
	assert(pNodeAnim->mNumScalingKeys > 0);

	for (uint i = 0 ; i < pNodeAnim->mNumScalingKeys - 1 ; i++) {
		if (AnimationTime < (float)pNodeAnim->mScalingKeys[i + 1].mTime) {
			return i;
		}
	}

	assert(0);

	return 0;
}

const aiNodeAnim* FKController::findNodeAnim(const aiAnimation* pAnimation, const QString NodeName)
{
	for (uint i = 0 ; i < pAnimation->mNumChannels ; i++) {
		const aiNodeAnim* pNodeAnim = pAnimation->mChannels[i];

		if (QString(pNodeAnim->mNodeName.data) == NodeName) {
			return pNodeAnim;
		}
	}

	return NULL;
}

void FKController::disableBoneChain( Bone* baseBone )
{
	m_disabledBones.clear();
	m_skeleton->makeBoneListFrom(baseBone, m_disabledBones);
}

void FKController::enableAllBones()
{
	m_disabledBones.clear();
}
