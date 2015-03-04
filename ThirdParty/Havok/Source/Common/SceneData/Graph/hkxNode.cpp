/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Graph/hkxNode.h>

hkxNode::~hkxNode()
{
	
}

hkxNode* hkxNode::findChildByName (const char* childName) const
{
	if( !childName )
	{
		return HK_NULL;
	}

	for (int i=0; i<m_children.getSize(); i++)
	{
		hkxNode* child = m_children[i];
		if( child->m_name && hkString::strCasecmp(child->m_name, childName)==0 )
		{
			return child;
		}
	}

	return HK_NULL;
}

hkxNode* hkxNode::findDescendantByName (const char* name) const
{
	if( !name )
	{
		return HK_NULL;
	}

	for (int i=0; i<m_children.getSize(); i++)
	{
		hkxNode* child = m_children[i];

		if( child->m_name && hkString::strCasecmp(child->m_name, name)==0 )
		{
			return child;
		}
		else
		{
			hkxNode* descendant = child->findDescendantByName(name);
			if (descendant != HK_NULL)
			{
				return descendant;
			}
		}
	}

	return HK_NULL;
}

hkResult hkxNode::getPathToNode (const hkxNode* theNode, hkArray<const hkxNode*>& pathOut) const
{
	pathOut.pushBack(this);

	if (this == theNode)
	{
		return HK_SUCCESS;
	}

	hkResult result = HK_FAILURE;

	for (int i=0; i<m_children.getSize(); i++)
	{
		result = m_children[i]->getPathToNode(theNode, pathOut);
		if (result==HK_SUCCESS) break;
	}

	if (result != HK_SUCCESS)
	{
		pathOut.popBack();
	}

	return result;
}


hkRefVariant hkxNode::findVariantByObject( const hkReferencedObject* object ) const
{
	if( m_object == object )
	{
		return m_object;
	}

	for( int i=0; i < m_children.getSize(); i++ )
	{
		hkRefVariant var = m_children[i]->findVariantByObject( object );
		if( var )
		{
			return var;
		}
	}

	return HK_NULL;
}


void hkxNode::replaceAllObjects( const hkReferencedObject* oldObject, const hkReferencedObject* newObject )
{
	if( m_object == oldObject )
	{
		m_object = newObject;
	}

	for( int i=0; i < m_children.getSize(); i++ )
	{
		m_children[i]->replaceAllObjects( oldObject, newObject );
	}
}

int hkxNode::getNumDescendants () const
{
	int numDescendants = m_children.getSize();

	for (int i=0; i<m_children.getSize(); i++)
	{
		numDescendants+=m_children[i]->getNumDescendants();
	}

	return numDescendants;
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
