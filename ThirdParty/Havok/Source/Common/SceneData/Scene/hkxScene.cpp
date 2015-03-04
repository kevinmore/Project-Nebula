/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Scene/hkxScene.h>

hkxScene::hkxScene()
: m_sceneLength (0),
  m_numFrames (0)
{
	m_appliedTransform.setIdentity();
}

hkxScene::~hkxScene()
{
	
}

hkxNode* hkxScene::findNodeByName (const char* name) const
{
	if( (!name) || (!m_rootNode) )
	{
		return HK_NULL;
	}
	else if ( m_rootNode->m_name && (hkString::strCasecmp(m_rootNode->m_name, name)==0))
	{
		return m_rootNode;
	}
	else
	{
		return m_rootNode->findDescendantByName(name);
	}
}

hkResult hkxScene::getFullPathToNode (const hkxNode* theNode, hkArray<const hkxNode*>& pathOut) const
{
	return m_rootNode->getPathToNode (theNode, pathOut);
}

hkResult hkxScene::getWorldFromNodeTransform (const hkxNode* theNode, hkMatrix4& worldFromNodeOut, int key) const
{
	hkInplaceArray<const hkxNode*,32> path;
	hkResult result = getFullPathToNode(theNode, path);

	if (result != HK_SUCCESS) return HK_FAILURE;

	worldFromNodeOut.setIdentity();

	for (int i=0; i<path.getSize(); i++)
	{
		int nodeKey = (key < path[i]->m_keyFrames.getSize() ) ? key : path[i]->m_keyFrames.getSize() - 1;
		const hkMatrix4& parentFromChild = path[i]->m_keyFrames[nodeKey];
		worldFromNodeOut.mul(parentFromChild);
	}

	return HK_SUCCESS;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
