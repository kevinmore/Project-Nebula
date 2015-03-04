/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeCache.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

hkTrackerTypeTreeCache::hkTrackerTypeTreeCache():	
	m_nodeFreeList(sizeof(Node), HK_ALIGN_OF(Node), 1024, &hkMemoryRouter::getInstance().heap())
{
	hkString::memSet(m_builtInTypes, 0, sizeof(m_builtInTypes));

	// Add the built in types
	for (int i = hkTrackerTypeTreeNode::TYPE_INT8; i <= hkTrackerTypeTreeNode::TYPE_VOID; i++)
	{
		m_builtInTypes[i] = newNode(Type(i));
	}
}

hkBool hkTrackerTypeTreeCache::getTypeExpressionTree(const char* name, const Node**nodeOut) const
{
	ExpressionTypeMap::Iterator iter = m_expressionTypeMap.findKey(name);
	if (m_expressionTypeMap.isValid(iter))
	{
		*nodeOut = m_expressionTypeMap.getValue(iter);
		return true;
	}
	else
	{
		*nodeOut = HK_NULL;
		return false;
	}
}


const hkTrackerTypeTreeCache::Node* hkTrackerTypeTreeCache::getTypeExpressionTree(const char* name) const
{
	return m_expressionTypeMap.getWithDefault(name, HK_NULL);
}		

void hkTrackerTypeTreeCache::setTypeExpressionTree(const char* name, const Node* node)
{
	HK_ASSERT(0x1bdc9979, node != HK_NULL);
	m_expressionTypeMap.insert(name, node);
}

hkTrackerTypeTreeCache::Node* hkTrackerTypeTreeCache::newNode(Type type)
{
	HK_ASSERT(0x3242a432, type != Node::TYPE_NAMED && type != Node::TYPE_ENUM && type != Node::TYPE_CLASS);

	return new (m_nodeFreeList.alloc()) hkTrackerTypeTreeNode(type);
}

const hkTrackerTypeTreeCache::Node* hkTrackerTypeTreeCache::newNamedNode(Type type, const hkSubString& name, hkBool allocName)
{
	const int len = name.length();
	hkLocalBuffer<char> buffer(len + 1);
	hkString::strNcpy(buffer.begin(), name.m_start, len);
	buffer[len] = 0;

	// I have to pass allocName as true - because it has to be allocated- as I'm passing in a temporary object
	return newNamedNode(type, buffer.begin(), true);
}

const hkTrackerTypeTreeCache::Node* hkTrackerTypeTreeCache::newNamedNode(Type type, const char* name, hkBool allocName)
{
	HK_ASSERT(0x3242a432, type == Node::TYPE_NAMED || type == Node::TYPE_ENUM || type == Node::TYPE_CLASS);

	// Look up and see if there is one which is not already in use
	{
		if( const Node* node = getNamedNode(name) )
		{
			// can only use it if it doesn't have stuff hanging off it
			if( node->m_next==HK_NULL )
			{
				// Change its type - if before it was just NAMED
				if (node->m_type != type && type != Node::TYPE_NAMED)
				{
					HK_ASSERT(0x243a4232, node->m_type == Node::TYPE_NAMED);
					Node* nonConstNode = const_cast<Node*>(node);
					nonConstNode->m_type = type;
				}

				return node;
			}
		}
	}

	Node* node = new (m_nodeFreeList.alloc()) hkTrackerTypeTreeNode(type);

	if (allocName)
	{
		name = newText(name);
	}

	node->m_name = name;
	// And to the named types
	m_namedTypeMap.insert(name, node);

	return node;
}

const hkTrackerTypeTreeCache::Node* hkTrackerTypeTreeCache::getNamedNode(const char* name) const
{
	return m_namedTypeMap.getWithDefault(name, HK_NULL);
}

const char* hkTrackerTypeTreeCache::newText(const char* text)
{
	TextMap::Iterator iter = m_textMap.findKey(text);
	if (m_textMap.isValid(iter))
	{
		return m_textMap.getKey(iter);
	}
	else
	{
		return m_textMap.insert(text, 1);
	}
}


void hkTrackerTypeTreeCache::clear()
{
	m_nodeFreeList.freeAllMemory();
	m_textMap.clear();
	m_expressionTypeMap.clear();
	m_namedTypeMap.clear();
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
