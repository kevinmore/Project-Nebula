/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Mesh/hkxVertexBuffer.h>

void hkxVertexBuffer::VertexData::clear()
{
	m_vectorData.clear();
	m_floatData.clear();
	m_uint32Data.clear();
	m_uint16Data.clear();
	m_uint8Data.clear();

	m_numVerts = 0;
	m_vectorStride = 0;
	m_floatStride = 0;
	m_uint32Stride = 0;
	m_uint16Stride = 0;
	m_uint8Stride = 0;
}

void hkxVertexBuffer::VertexData::setSize(int n)
{
	m_numVerts = n;
	m_uint8Data.setSize(n*m_uint8Stride);
	m_uint16Data.setSize(n*m_uint16Stride/sizeof(hkUint16));
	m_uint32Data.setSize(n*m_uint32Stride/sizeof(hkUint32));
	m_floatData.setSize(n*m_floatStride/sizeof(hkFloat32));
	m_vectorData.setSize(n*m_vectorStride/sizeof(hkFloat32));  // 4 of them always
}

void hkxVertexBuffer::VertexData::expandBy(int n)
{
	m_numVerts += n;
	m_uint8Data.expandBy(n*m_uint8Stride);
	m_uint16Data.expandBy(n*m_uint16Stride/sizeof(hkUint16));
	m_uint32Data.expandBy(n*m_uint32Stride/sizeof(hkUint32));
	m_floatData.expandBy(n*m_floatStride/sizeof(hkFloat32));
	m_vectorData.expandBy(n*m_vectorStride/sizeof(hkFloat32));  // 4 of them always
}


hkxVertexBuffer::hkxVertexBuffer( hkFinishLoadedObjectFlag f )
: hkReferencedObject(f), m_data(f), m_desc(f)
{

}

void hkxVertexBuffer::setNumVertices(int n, const hkxVertexDescription& format) 
{
	if ((n == (int)m_data.m_numVerts) && (format == m_desc))
		return; // already ok

	m_desc.m_decls.clear();
	m_data.clear();

	for (int i = 0; i < format.m_decls.getSize(); ++i)
	{
		const hkxVertexDescription::ElementDecl& fDecl = format.m_decls[i];
		hkxVertexDescription::ElementDecl& newDecl = m_desc.m_decls.expandOne();
		newDecl.m_type = fDecl.m_type;
		newDecl.m_usage = fDecl.m_usage;
		newDecl.m_numElements = fDecl.m_numElements;
		newDecl.m_hint = fDecl.m_hint;

		switch ( fDecl.m_type )
		{
			case hkxVertexDescription::HKX_DT_UINT8:
				newDecl.m_byteOffset = m_data.m_uint8Stride;
				m_data.m_uint8Stride += sizeof(hkUint8) * fDecl.m_numElements; 
				break;
			case hkxVertexDescription::HKX_DT_INT16:
				newDecl.m_byteOffset = m_data.m_uint16Stride;
				m_data.m_uint16Stride += sizeof(hkUint16) * fDecl.m_numElements; 
				break;
			case hkxVertexDescription::HKX_DT_UINT32:
				newDecl.m_byteOffset = m_data.m_uint32Stride;
				m_data.m_uint32Stride += sizeof(hkUint32) * fDecl.m_numElements; 
				break;
			case hkxVertexDescription::HKX_DT_FLOAT:
				{
					if(fDecl.m_numElements == 3 || fDecl.m_numElements == 4)
					{
						newDecl.m_byteOffset = m_data.m_vectorStride;
						m_data.m_vectorStride += (4*sizeof(hkFloat32));  // 4 of them always					
					}
					else if(fDecl.m_numElements < 3 )
					{
						newDecl.m_byteOffset = m_data.m_floatStride;
						m_data.m_floatStride += sizeof(hkFloat32) * fDecl.m_numElements; 					
					}
				}
				break;
			default: break;
		}
	}

	for (int ii = 0; ii < m_desc.m_decls.getSize(); ++ii)
	{
		const hkxVertexDescription::ElementDecl& fDecl = format.m_decls[ii];

		hkxVertexDescription::ElementDecl& cDecl = m_desc.m_decls[ii];
		switch ( cDecl.m_type )
		{
		case hkxVertexDescription::HKX_DT_UINT8:
			cDecl.m_byteStride = m_data.m_uint8Stride;
			break;
		case hkxVertexDescription::HKX_DT_INT16:
			cDecl.m_byteStride = m_data.m_uint16Stride;
			break;
		case hkxVertexDescription::HKX_DT_UINT32:
			cDecl.m_byteStride = m_data.m_uint32Stride;
			break;
		case hkxVertexDescription::HKX_DT_FLOAT:
			{
				if(fDecl.m_numElements == 3 || fDecl.m_numElements == 4)
				{	
					cDecl.m_byteStride = m_data.m_vectorStride;  // 4 of them always
				}
				else if(fDecl.m_numElements < 3 )
				{	
					cDecl.m_byteStride = m_data.m_floatStride;
				}
			}
			break;
		default: break;
		}
	}

	m_data.m_numVerts = n;
	m_data.m_uint8Data.setSize(n*m_data.m_uint8Stride);
	m_data.m_uint16Data.setSize(n*m_data.m_uint16Stride/sizeof(hkUint16));
	m_data.m_uint32Data.setSize(n*m_data.m_uint32Stride/sizeof(hkUint32));
	m_data.m_floatData.setSize(n*m_data.m_floatStride/sizeof(hkFloat32));
	m_data.m_vectorData.setSize(n*m_data.m_vectorStride/sizeof(hkFloat32)); // 4 of them always

}

void hkxVertexBuffer::expandNumVertices(int n)
{
	HK_ASSERT(0x1a3e32e, n >= 0);
	m_data.expandBy(n);
}

void* hkxVertexBuffer::getVertexDataPtr(const hkxVertexDescription::ElementDecl& elem)
{
	void* base;
	switch ( elem.m_type )
	{
		case hkxVertexDescription::HKX_DT_UINT8:
			base = m_data.m_uint8Data.begin();
			break;
		case hkxVertexDescription::HKX_DT_INT16:
			base = m_data.m_uint16Data.begin();
			break;
		case hkxVertexDescription::HKX_DT_UINT32:
			base = m_data.m_uint32Data.begin();
			break;
		case hkxVertexDescription::HKX_DT_FLOAT:
			{
				if(elem.m_numElements == 3 || elem.m_numElements == 4)
				{
					base = m_data.m_vectorData.begin();  // 4 of them always
				}
				else if(elem.m_numElements < 3 )
				{
					base = m_data.m_floatData.begin();
				}
				else
				{
					HK_WARN(0x234fcf, "hkxVertexBuffer::getVertexDataPtr : Invalid numElements for ElementDecl" );
					return HK_NULL;
				}
			}
			break;
		default: 
			HK_WARN(0x234fce, "hkxVertexBuffer::getVertexDataPtr : Invalid ElementDecl" );
			return HK_NULL;
	}
	return hkAddByteOffset(base, elem.m_byteOffset);
}

const void* hkxVertexBuffer::getVertexDataPtr(const hkxVertexDescription::ElementDecl& elem) const
{
	return const_cast<hkxVertexBuffer*>(this)->getVertexDataPtr(elem);
}

void hkxVertexBuffer::copy( const hkxVertexBuffer& other, bool resize)
{
	bool sameLayout = m_desc == other.m_desc;

	if (resize)
	{
		setNumVertices( other.m_data.m_numVerts, m_desc	);
	}

	if (sameLayout)
	{
		int numVerts = hkMath::min2<hkUint32>( other.m_data.m_numVerts, m_data.m_numVerts );

		if (m_data.m_uint8Stride > 0)
			hkString::memCpy((char*)m_data.m_uint8Data.begin(), (const char*)other.m_data.m_uint8Data.begin(), m_data.m_uint8Stride * numVerts); 

		if (m_data.m_uint16Stride > 0)
			hkString::memCpy((char*)m_data.m_uint16Data.begin(), (const char*) other.m_data.m_uint16Data.begin(), m_data.m_uint16Stride * numVerts); 

		if (m_data.m_uint32Stride > 0)
			hkString::memCpy((char*)m_data.m_uint32Data.begin(), (const char*)other.m_data.m_uint32Data.begin(), m_data.m_uint32Stride * numVerts); 

		if (m_data.m_floatStride > 0)
			hkString::memCpy((char*)m_data.m_floatData.begin(), (const char*)other.m_data.m_floatData.begin(), m_data.m_floatStride * numVerts); 

		if (m_data.m_vectorStride > 0)
			hkString::memCpy((char*)m_data.m_vectorData.begin(), (const char*)other.m_data.m_vectorData.begin(), m_data.m_vectorStride * numVerts); 		
	}
	else
	{
		// format not the same, have to do per elem (slow..)
		for (hkUint32 v=0; v < other.m_data.m_numVerts; ++v)
		{
			copyVertex(other, v, v );
		}
	}
}

void hkxVertexBuffer::copyVertex( const hkxVertexBuffer& other, int vertFrom, int vertTo)
{
	bool sameLayout = m_desc == other.m_desc;
	int dataUsageCount[(int)hkxVertexDescription::HKX_DU_USERDATA + 1];
	hkString::memSet(dataUsageCount, 0, ((int)hkxVertexDescription::HKX_DU_USERDATA + 1)*sizeof(int) );
	if ((vertTo < (int)m_data.m_numVerts) && (vertFrom < (int)other.m_data.m_numVerts))
	{
		if (sameLayout)
		{
			if (m_data.m_uint8Stride > 0)
				hkString::memCpy((char*)m_data.m_uint8Data.begin() + (m_data.m_uint8Stride * vertTo), (const char*)other.m_data.m_uint8Data.begin() + (m_data.m_uint8Stride * vertFrom), m_data.m_uint8Stride); 

			if (m_data.m_uint16Stride > 0)
				hkString::memCpy((char*)m_data.m_uint16Data.begin() + (m_data.m_uint16Stride * vertTo), (const char*) other.m_data.m_uint16Data.begin() + (m_data.m_uint16Stride * vertFrom), m_data.m_uint16Stride); 

			if (m_data.m_uint32Stride > 0)
				hkString::memCpy((char*)m_data.m_uint32Data.begin() + (m_data.m_uint32Stride * vertTo), (const char*)other.m_data.m_uint32Data.begin() + (m_data.m_uint32Stride * vertFrom), m_data.m_uint32Stride); 

			if (m_data.m_floatStride > 0)
				hkString::memCpy((char*)m_data.m_floatData.begin() + (m_data.m_floatStride * vertTo), (const char*)other.m_data.m_floatData.begin() + (m_data.m_floatStride * vertFrom), m_data.m_floatStride ); 

			if (m_data.m_vectorStride > 0)
				hkString::memCpy((char*)m_data.m_vectorData.begin() + (m_data.m_vectorStride * vertTo), (const char*)other.m_data.m_vectorData.begin() + (m_data.m_vectorStride * vertFrom), m_data.m_vectorStride); 		
			
		}
		else
		{
			for (int di=0; di < m_desc.m_decls.getSize(); ++di)
			{
				const hkxVertexDescription::ElementDecl& toEd = m_desc.m_decls[di];
				const hkxVertexDescription::ElementDecl* fromEd = other.m_desc.getElementDecl( toEd.m_usage, dataUsageCount[toEd.m_usage] );
				if (fromEd && (toEd.m_type == fromEd->m_type))
				{
					dataUsageCount[toEd.m_usage]++;

					// copy
					char* toData = (char*)getVertexDataPtr( toEd );
					char* fromData = (char*)other.getVertexDataPtr( *fromEd );

					int size = m_desc.getByteSizeForType( toEd.m_type, fromEd->m_numElements );
					HK_ASSERT(0x3a6e6cae, size > 0);
					hkString::memCpy( toData + (vertTo*toEd.m_byteStride), fromData + (vertFrom*fromEd->m_byteStride), size );
				}
				
				HK_WARN_ON_DEBUG_IF((fromEd && (toEd.m_type != fromEd->m_type)), 0xefe34ce, "copyVertex does not support differemt base types.");
			}
		}
	}
	else
	{
		HK_WARN(0xefe34ce, "hkxVertexBuffer::copyVertex: 'to' or 'from' index of of range" );
	}
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
