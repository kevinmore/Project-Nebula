/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

// this
#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeNode.h>

hkTrackerTypeTreeNode::hkTrackerTypeTreeNode(Type type):
	m_type(type)
{
	m_name.empty();
	m_dimension = 0;
	m_contains = HK_NULL;
	m_next = HK_NULL;
}

hkBool hkTrackerTypeTreeNode::isNamedType() const
{
	return m_type == TYPE_NAMED || m_type == TYPE_ENUM || m_type == TYPE_CLASS || m_type == TYPE_CLASS_TEMPLATE;
}
			
hkBool hkTrackerTypeTreeNode::isNamedType(const char* name) const
{
	return isNamedType() && m_name == name;
}

void hkTrackerTypeTreeNode::dumpType(hkOstream& stream) const
{
	switch (m_type)
	{
		case TYPE_UNKNOWN:
		{
			stream << "(Unknown)"; 
			break;
		}
		case TYPE_POINTER:
		{
			m_contains->dumpType( stream);
			stream << "*";
			break;
		}
		case TYPE_REFERENCE:
		{
			m_contains->dumpType(stream);
			stream << "&";
			break;
		}
		case TYPE_CLASS_TEMPLATE:
		{
			stream << m_name;
			stream << '<';
			const hkTrackerTypeTreeNode* cur = m_contains;

			while (cur)
			{
				if (cur != m_contains)
				{
					stream << ',';

				}
				cur->dumpType(stream);
				// Next
				cur = cur->m_next;
			}
			stream << '>';
			break;
		}
		case TYPE_INT8:
		{
			stream << "int8";
			break;
		}
		case TYPE_INT16:
		{
			stream << "int16";
			break;
		}
		case TYPE_INT32:
		{
			stream << "int32";
			break;
		}
		case TYPE_INT64:
		{
			stream << "int64";
			break;
		}
		case TYPE_UINT8:
		{
			stream << "uint8";
			break;
		}
		case TYPE_UINT16:
		{
			stream << "uint16";
			break;
		}
		case TYPE_UINT32:
		{
			stream << "uint32";
			break;
		}
		case TYPE_UINT64:
		{
			stream << "uint64";
			break;
		}
		case TYPE_FLOAT32:
		{
			stream << "float32";
			break;
		}
		case TYPE_FLOAT64:
		{
			stream << "float64";
			break;
		}
		case TYPE_BOOL:
		{
			stream << "bool";
			break;
		}
		case TYPE_VOID:
		{
			stream << "void";
			break;
		}
		case TYPE_ENUM:
		case TYPE_CLASS:
		case TYPE_NAMED:
		{
			stream << m_name;
			break;
		}
		case TYPE_ARRAY:
		{
			m_contains->dumpType(stream);
			stream << "[" << m_dimension << "]";
			break;
		}
		case TYPE_INT_VALUE:
		{
			stream << m_dimension;
			break;
		}
		default: break;
	}
}

/* static */void HK_CALL hkTrackerTypeTreeNode::dumpType(const hkTrackerTypeTreeNode* node, hkOstream& stream)
{
	if (!node)
	{
		stream << "(Unknown)";
		return;
	}

	node->dumpType(stream);
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
