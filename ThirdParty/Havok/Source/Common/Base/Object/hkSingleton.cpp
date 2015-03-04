/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Container/StringMap/hkStringMap.h>

struct hkSingletonInitNode* hkSingletonInitList;

void hkSingletonInitNode::populate(hkSingletonInitNode* dstBase, const hkSingletonInitNode* srcBase)
{
	hkStringMap<const hkSingletonInitNode*> srcNodeFromName;
	for( const hkSingletonInitNode* srcCur = srcBase; srcCur != HK_NULL; srcCur = srcCur->m_next )
	{
		srcNodeFromName.insert( srcCur->m_name, srcCur );
	}
	for( const hkSingletonInitNode* dstCur = dstBase; dstCur != HK_NULL; dstCur = dstCur->m_next )
	{
		if( const hkSingletonInitNode* n = srcNodeFromName.getWithDefault(dstCur->m_name, HK_NULL) )
		{
			*dstCur->m_value = *n->m_value;
			if (*dstCur->m_value) 
			{
				((hkReferencedObject*)(*dstCur->m_value))->addReferenceLockUnchecked();
			}
		}
	}
}

void hkSingletonInitNode::depopulate(hkSingletonInitNode* dstBase, const hkSingletonInitNode* srcBase)
{
	hkStringMap<const hkSingletonInitNode*> srcNodeFromName;
	for( const hkSingletonInitNode* srcCur = srcBase; srcCur != HK_NULL; srcCur = srcCur->m_next )
	{
		srcNodeFromName.insert( srcCur->m_name, srcCur );
	}
	for( const hkSingletonInitNode* dstCur = dstBase; dstCur != HK_NULL; dstCur = dstCur->m_next )
	{
		if( const hkSingletonInitNode* n = srcNodeFromName.getWithDefault(dstCur->m_name, HK_NULL) )
		{
			if (*dstCur->m_value) 
			{
				((hkReferencedObject*)(*dstCur->m_value))->removeReferenceLockUnchecked();
			}
			*dstCur->m_value = HK_NULL;
		}
	}
}

hkSingletonInitNode* hkSingletonInitNode::findByName(const char* name)
{
	if (m_name && (hkString::strCmp(name, m_name) == 0))
	{
		return this;
	}
	if (m_next) return m_next->findByName(name);
	return HK_NULL;
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
