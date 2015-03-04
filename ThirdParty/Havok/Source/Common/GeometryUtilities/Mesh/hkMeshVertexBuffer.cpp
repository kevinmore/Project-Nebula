/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Common/Base/Container/String/hkStringBuf.h>

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! hkVertexFormat::Element !!!!!!!!!!!!!!!!!!!!*/

void hkVertexFormat::Element::getText(hkStringBuf& buf) const
{
	buf.printf("%s(%i) %s(%i)", hkVertexFormat::s_usageText[m_usage], int(m_subUsage), hkVertexFormat::s_typeText[m_dataType], int(m_numValues));
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! hkVertexFormat !!!!!!!!!!!!!!!!!!!!!!!!!!*/

/* static */const hkUint8 hkVertexFormat::s_dataTypeToSize[] =
{
    0,
    (hkUint8)sizeof(hkInt8),
    (hkUint8)sizeof(hkUint8),
    (hkUint8)sizeof(hkInt16),
    (hkUint8)sizeof(hkUint16),
    (hkUint8)sizeof(hkInt32),
    (hkUint8)sizeof(hkUint32),
    (hkUint8)sizeof(hkUint8),       // TYPE_UINT8_DWORD (the num elements is still the amount of bytes - so its really about byte order)
    (hkUint8)sizeof(hkUint32),      // TYPE_ARGB32
    (hkUint8)sizeof(hkUint16),      // TYPE_FLOAT16
    (hkUint8)sizeof(hkFloat32),
	(hkUint8)sizeof(hkVector4),
    0
};

/* static */const char* const hkVertexFormat::s_typeText[] =
{
	"NONE",		// TYPE_NONE = 0,
	"INT8",		// TYPE_INT8,
	"UINT8",	// TYPE_UINT8,
	"INT16",	// TYPE_INT16,
	"UINT16",	// TYPE_UINT16,
	"INT32",	// TYPE_INT32,
	"UINT32",	// TYPE_UINT32,
	"UINT8_DWORD",	// TYPE_UINT8_DWORD,
	"ARGB32",	// TYPE_ARGB32,						///< A color packed into 4 bytes (interpreted as vector4 , for getting/setting element arrays)
	"FLOAT16", 	// TYPE_FLOAT16,
	"FLOAT32",	// TYPE_FLOAT32,
	"VECTOR4",	// TYPE_VECTOR4,	
};

/* static */const char* const hkVertexFormat::s_usageText[] = 
{
	"NONE",			// USAGE_NONE = 0,
	"POSITION",		// USAGE_POSITION = 1,					
	"NORMAL",		// USAGE_NORMAL,						
	"COLOR",		// USAGE_COLOR,						
	"TANGENT",		// USAGE_TANGENT,						
	"BINORMAL",		// USAGE_BINORMAL,						
	"BLEND_MATRIX_INDEX",	// USAGE_BLEND_MATRIX_INDEX,			
	"BLEND_WEIGHTS",		// USAGE_BLEND_WEIGHTS,				
	"BLEND_WEIGHTS_LAST_IMPLIED",	// USAGE_BLEND_WEIGHTS_LAST_IMPLIED,	
	"TEX_COORD",	// USAGE_TEX_COORD,					
	"POINT_SIZE",	// USAGE_POINT_SIZE,					
	"USER",			// USAGE_USER,							
};


hkVertexFormat::hkVertexFormat()
	: m_numElements(0) 
{
}

hkVertexFormat::hkVertexFormat(const hkVertexFormat& rhs) 
{ 
	set(rhs); 
}

void hkVertexFormat::operator=(const hkVertexFormat& rhs) 
{ 
	set(rhs); 
}

bool hkVertexFormat::operator==(const hkVertexFormat& rhs) const
{
    HK_ASSERT(0x3424308, isCanonicalOrder() && rhs.isCanonicalOrder());

    if (m_numElements != rhs.m_numElements)
    {
        return false;
    }

    const int numElements = m_numElements;
    for (int i = 0; i < numElements; i++)
    {
        if (m_elements[i] != rhs.m_elements[i])
        {
            return false;
        }
    }
    return true;
}

void hkVertexFormat::set(const hkVertexFormat& rhs)
{
    m_numElements = rhs.m_numElements;
    const int numElements = rhs.m_numElements;

    for (int i = 0; i < numElements; i++)
    {
        m_elements[i] = rhs.m_elements[i];
    }
}


int hkVertexFormat::findNextSubUsage(ComponentUsage usage) const
{
    int max = 0;
    const int numElements = m_numElements;
    for (int i = 0; i < numElements; i++)
    {
        const Element& ele = m_elements[i];
        if (ele.m_usage == usage)
        {
            if (ele.m_subUsage + 1 > max)
            {
                max = ele.m_subUsage + 1;
            }
        }
    }
    return max;
}

int hkVertexFormat::findElementIndex(ComponentUsage usage, int subUsage) const
{
    const int numElements = m_numElements;
    for (int i = 0; i < numElements; i++)
    {
        const Element& ele = m_elements[i];
        if (ele.m_usage == usage && ele.m_subUsage == subUsage)
        {
            return i;
        }
    }
    return -1;
}

bool hkVertexFormat::isCanonicalOrder() const
{
    const int numElements = m_numElements;
    for (int i = 1; i < numElements; i++)
    {
        const Element& prev = m_elements[i - 1];
        const Element& comp = m_elements[i];

        if (prev.m_usage > comp.m_usage)
        {
            return false;
        }
        if (prev.m_usage < comp.m_usage)
        {
            continue;
        }

        if (prev.m_subUsage > comp.m_subUsage)
        {
            return false;
        }
        if (prev.m_subUsage == comp.m_subUsage)
        {
            // Its badly formed - there is more than one element with same usage/subUsage
            return false;
        }
    }

    return true;
}

HK_FORCE_INLINE static bool hkVertexFormat_orderComponents(const hkVertexFormat::Element& a, const hkVertexFormat::Element& b)
{
    return (a.m_usage < b.m_usage) || ( (a.m_usage == b.m_usage) && (a.m_subUsage < b.m_subUsage) );
}

void hkVertexFormat::makeCanonicalOrder()
{
    if (isCanonicalOrder())
    {
        return;
    }

    hkSort(m_elements, m_numElements, hkVertexFormat_orderComponents);

    HK_ASSERT(0x277abd2b, isCanonicalOrder());
}

void hkVertexFormat::addElement(ComponentUsage usage, ComponentType type, int numValues, int flags)
{
    if (m_numElements >= MAX_ELEMENTS)
    {
        HK_ASSERT(0x7bba5469, !"All element entries are used");
        return;
    }

    int subUsage = findNextSubUsage(usage);


    Element& ele = m_elements[m_numElements++];

    ele.m_dataType = type;
    ele.m_numValues = hkUint8(numValues);
    ele.m_usage = usage;
    ele.m_subUsage = hkUint8(subUsage);
    ele.m_flags = hkUint8(flags);
}

void hkVertexFormat::addElement(const Element& element)
{
	if (m_numElements >= MAX_ELEMENTS)
	{
		HK_ASSERT(0x7bba5468, !"All element entries are used");
		return;
	}

	if (findElementIndex(element.m_usage, element.m_subUsage) >= 0)
	{
		HK_ASSERT(0x424324, !"Element with same usage/subUsage already in format");
		return;
	}

	m_elements[m_numElements++] = element;
}

hkVertexFormat::SharingType hkVertexFormat::calculateSharingType() const
{
    const int numElements = m_numElements;
    int numNotShared = 0;

    for (int i = 0; i < numElements; i++)
    {
        const Element& ele = m_elements[i];
        if (ele.m_flags.anyIsSet(FLAG_NOT_SHARED))
        {
            numNotShared ++;
        }
    }
    if (numNotShared == 0)
    {
        return SHARING_ALL_SHARED;
    }
    if (numNotShared == numElements)
    {
        return SHARING_ALL_NOT_SHARED;
    }
    return SHARING_MIXTURE;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                          LockedVertices

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

int hkMeshVertexBuffer::LockedVertices::findBufferIndex(hkVertexFormat::ComponentUsage usage, int subUsage) const
{
    const int numBuffers = m_numBuffers;
    for (int i = 0; i < numBuffers; i++)
    {
        const hkVertexFormat::Element& ele = m_buffers[i].m_element;

        if (ele.m_usage == usage && ele.m_subUsage == subUsage)
        {
            return i;
        }
    }
    return -1;
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
