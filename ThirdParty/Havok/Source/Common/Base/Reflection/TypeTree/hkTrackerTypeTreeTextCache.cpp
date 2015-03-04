/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>

// this
#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeTextCache.h>

int hkTrackerTypeTreeTextCache::getTypeIndex(const Node* type)
{
	if (type == HK_NULL)
	{
		return -1;
	}

	int index = m_map.getWithDefault(type, -1);
	if (index >= 0)
	{
		return index;
	}


	// Add it
	char buffer[256];
	hkOstream stream(buffer, HK_COUNT_OF(buffer), true);

	switch (type->m_type)
	{
		case hkTrackerTypeTreeNode::TYPE_NAMED:
		{
			break;
		}
		case hkTrackerTypeTreeNode::TYPE_ENUM:
		{
			stream << "enum ";
			break;
		}
		case hkTrackerTypeTreeNode::TYPE_CLASS_TEMPLATE:
		case hkTrackerTypeTreeNode::TYPE_CLASS:
		{
			stream << "class ";
			break;
		}
		default: break;
	}

	type->dumpType(stream);

	int len = hkString::strLen(buffer);

	index = m_typeNames.getSize();
	char* dst = (char*)m_typeNames.expandBy(len + 1);
	hkString::strCpy(dst, buffer);

	// Add the new one
	m_map.insert(type, index);

	return index;
}

const char* hkTrackerTypeTreeTextCache::getTypeText(const Node* type)
{
	int index = getTypeIndex(type);
	if (index < 0)
	{
		return HK_NULL;
	}

	// Get the type name
	return &m_typeNames[index];
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
