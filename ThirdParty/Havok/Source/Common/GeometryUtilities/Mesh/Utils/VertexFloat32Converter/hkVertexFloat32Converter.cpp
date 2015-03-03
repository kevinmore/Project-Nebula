/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexFloat32Converter/hkVertexFloat32Converter.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Base/Math/Vector/hkIntVector.h>

/* static */void HK_CALL hkVertexFloat32Converter::convertPositionsInternal(const hkAabb& aabb, hkSimdRealParameter weightIn, hkVector4* positions, int numPos)
{
	hkVector4 offset; offset = aabb.m_min;
	hkVector4 localToWorldScale; localToWorldScale.setSub(aabb.m_max, aabb.m_min);
	hkVector4 worldToLocalScale;
	worldToLocalScale.setReciprocal(localToWorldScale); 
	worldToLocalScale.setComponent<3>(hkSimdReal_1);
	worldToLocalScale.mul(weightIn);

	for (int i = 0; i < numPos; i++)
	{
		hkVector4 p = positions[i];
		p.sub(offset);
		p.mul(worldToLocalScale);
		positions[i] = p;
	}
}

void hkVertexFloat32Converter::init(const hkVertexFormat& format, const hkAabb& aabb, bool unitScalePosition)
{
	m_positionIndex = -1;
	m_entries.clear();

	// Set the AABB
	m_positionAabb = aabb;

	m_offset = aabb.m_min;

	if ( unitScalePosition )
	{
		m_localToWorldScale.setSub(aabb.m_max, aabb.m_min);
		m_worldToLocalScale.setReciprocal(m_localToWorldScale); 
		m_worldToLocalScale.setComponent<3>(hkSimdReal_1);
	}
	else
	{
		m_localToWorldScale = hkVector4::getConstant<HK_QUADREAL_1>();
		m_worldToLocalScale = m_localToWorldScale;
	}
	//
	int offsets[hkVertexFormat::MAX_ELEMENTS];
	hkMemoryMeshVertexBuffer::calculateElementOffsets(format, offsets);

	const int numEles = format.m_numElements;

	int numReals = 0;
	for (int i = 0; i < numEles; i++)
	{
		const hkVertexFormat::Element& ele = format.m_elements[i];

		const int numValues = ele.m_numValues;

		SrcType srcType = SRC_TYPE_UNKNOWN;
		DstType dstType = DST_TYPE_UNKNOWN;

		int numElementReals = 0;
		switch (ele.m_dataType)
		{
			case hkVertexFormat::TYPE_FLOAT32:
			{
				srcType = SRC_TYPE_FLOAT32_GENERAL;
				dstType = DST_TYPE_FLOAT32_GENERAL;

				numElementReals = numValues;

				switch (ele.m_usage)
				{
					case hkVertexFormat::USAGE_POSITION:
					{
						if (ele.m_numValues == 3)
						{
							srcType = SRC_TYPE_FLOAT32_POSITION;
							dstType = DST_TYPE_FLOAT32_POSITION;
						}

						if (ele.m_subUsage == 0)
						{
							// This is the position index
							m_positionIndex = m_entries.getSize();
						}
						break;
					}
					case hkVertexFormat::USAGE_BINORMAL:
					case hkVertexFormat::USAGE_NORMAL:
					case hkVertexFormat::USAGE_TANGENT:
					{
						if (ele.m_numValues == 3)
						{
							dstType = DST_TYPE_FLOAT32_NORMAL;
						}
						break;
					}
					case hkVertexFormat::USAGE_COLOR:
					{
						if (ele.m_numValues >= 1 && ele.m_numValues <= 4)
						{
							dstType = DstType(DST_TYPE_FLOAT32_CLAMP1 + (ele.m_numValues - 1));
						}
						break;
					}
					default: break;
				}
				break;
			}
			case hkVertexFormat::TYPE_ARGB32:
			{
				numElementReals = numValues * 4;

				srcType = SRC_TYPE_ARGB32;
				dstType = DST_TYPE_ARGB32;
				break;
			}
			default:
			{
				HK_ASSERT(0x3244a3a2, !"Unhandled type");
			}
		}

		Entry& entry = m_entries.expandOne();

		entry.m_srcType = srcType;
		entry.m_dstType = dstType;
		entry.m_numValues = numValues;
		entry.m_realOffset = numReals;
		entry.m_offset = offsets[i];

		entry.m_weight = hkSimdReal_1;
		entry.m_recipWeight = hkSimdReal_1;
		
		numReals += numElementReals;
	}

	m_numReals = numReals;
	m_format = format;
}

int hkVertexFloat32Converter::findElementOffset(hkVertexFormat::ComponentUsage usage, int subUsage) const
{
	int index = m_format.findElementIndex(usage, subUsage);
	if (index < 0) 
	{
		return -1;
	}
	return m_entries[index].m_realOffset;
}

void hkVertexFloat32Converter::setWeight(hkVertexFormat::ComponentUsage usage, int subUsage, hkSimdRealParameter weight)
{
	const int index = m_format.findElementIndex(usage, subUsage);
	HK_ASSERT(0x32423432, index >= 0);

	Entry& entry = m_entries[index];

	entry.m_weight = weight;
	entry.m_recipWeight.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO>(weight);
}

void hkVertexFloat32Converter::getWeight(hkVertexFormat::ComponentUsage usage, int subUsage, hkSimdReal& weight) const
{
	const int index = m_format.findElementIndex(usage, subUsage);
	weight = (index >= 0) ? m_entries[index].m_weight : hkSimdReal_1;
}

void hkVertexFloat32Converter::convertPositionToInternal(hkVector4Parameter in, hkVector4& out) const
{
	const Entry& posEntry = m_entries[m_positionIndex];

	out.setSub(in, m_offset);
	out.mul(m_worldToLocalScale);
	out.mul(posEntry.m_weight);
}

void hkVertexFloat32Converter::convertInternalToPosition(hkVector4Parameter in, hkVector4& out) const
{
	const Entry& posEntry = m_entries[m_positionIndex];

	out.setMul(in, posEntry.m_recipWeight);
	out.mul(m_localToWorldScale);
	out.add(m_offset);
}

void hkVertexFloat32Converter::convertVertexToFloat32(const void* vertexIn, hkFloat32* reals) const
{
	const hkUint8* vertex = (const hkUint8*) vertexIn;

	const Entry* cur = m_entries.begin();
	const Entry* end = m_entries.end();

	for (; cur < end; cur++)
	{
		switch (cur->m_srcType)
		{
			case SRC_TYPE_FLOAT32_GENERAL:
			{
				const hkFloat32* srcV = (const hkFloat32*)(vertex + cur->m_offset);
				const hkFloat32* endV = srcV + cur->m_numValues;
				hkFloat32* dstV = reals + cur->m_realOffset;
				const hkSimdReal weight = cur->m_weight;

				while ( srcV < endV ) 
				{
					hkSimdReal s; 
					s.load<1>(srcV++);
					(s * weight).store<1>(dstV++);
				}
				break;
			}
			case SRC_TYPE_FLOAT32_POSITION:
			{
				// If its position it has to be float32, 3
				hkVector4 p;
				const hkFloat32* v = (const hkFloat32*)(vertex + cur->m_offset);
				p.load<3,HK_IO_NATIVE_ALIGNED>(v);
				p.zeroComponent<3>();
				p.sub(m_offset);
				p.mul(m_worldToLocalScale);
				p.mul(cur->m_weight);
				// Now need to store result
				p.store<3,HK_IO_NATIVE_ALIGNED>(reals + cur->m_realOffset);
				break;
			}
			case SRC_TYPE_ARGB32:
			{
				hkSimdReal scale; scale.setMul(cur->m_weight, hkSimdReal_Inv_255);

				const hkUint32* src = (const hkUint32*)(vertex + cur->m_offset);
				const hkUint32* srcEnd = src + cur->m_numValues;

				while (src < srcEnd)
				{
					const hkUint32 packed = *src++;
					hkVector4 v; 
					hkIntVector iv; iv.set(packed & 0xff, (packed >> 8)& 0xff, (packed >> 16)& 0xff, (packed >> 24)& 0xff);
					iv.convertS32ToF32(v);
					v.mul(scale);
					v.store<4,HK_IO_NATIVE_ALIGNED>(reals + cur->m_realOffset);
				}
				break;
			}
			default:
			{
				// Can't handle this
				HK_ASSERT(0x32423423, !"Unhandled type");
			}
		}
	}
}

int hkVertexFloat32Converter::countVertexToFloat32() const
{
	const Entry* cur = m_entries.begin();
	const Entry* end = m_entries.end();

	int floatCount = 0;
	for (; cur < end; cur++)
	{
		switch (cur->m_srcType)
		{
		case SRC_TYPE_FLOAT32_GENERAL:
			{
				floatCount = hkMath::max2(floatCount, cur->m_realOffset + cur->m_numValues);
				break;
			}
		case SRC_TYPE_FLOAT32_POSITION:
			{
				// If its position it has to be float32, 3
				floatCount = hkMath::max2(floatCount, cur->m_realOffset + 3);
				break;
			}
		case SRC_TYPE_ARGB32:
			{
				floatCount = hkMath::max2(floatCount, cur->m_realOffset + 4);
				break;
			}
		default:
			{
				// Can't handle this
				HK_ASSERT(0x32423423, !"Unhandled type");
			}
		}
	}
	return floatCount;
}

void hkVertexFloat32Converter::convertFloat32ToVertex(const hkFloat32* reals, void* vertexOut) const
{
	hkUint8* vertex = (hkUint8*) vertexOut;

	hkVector4 ones = hkVector4::getConstant<HK_QUADREAL_1>();
	hkVector4 zeros; zeros.setZero();
	hkVector4 magic; magic.setAll(0x800000);
	hkVector4 all255 = hkVector4::getConstant<HK_QUADREAL_255>();

	const Entry* cur = m_entries.begin();
	const Entry* end = m_entries.end();

	for (; cur != end; cur++)
	{
		switch (cur->m_dstType)
		{
			case DST_TYPE_FLOAT32_POSITION:
			{
				// If its position it has to be float32, 3
				hkVector4 p;
				p.load<3,HK_IO_NATIVE_ALIGNED>(reals + cur->m_realOffset);
				p.mul(cur->m_recipWeight);
				p.mul(m_localToWorldScale);
				p.add(m_offset);
				p.store<3,HK_IO_NATIVE_ALIGNED>((hkFloat32*)(vertex + cur->m_offset));
				break;
			}
			case DST_TYPE_FLOAT32_NORMAL:
			{
				hkVector4 p;
				p.load<3,HK_IO_NATIVE_ALIGNED>(reals + cur->m_realOffset);
				p.normalize<3>();
				p.store<3,HK_IO_NATIVE_ALIGNED>((hkFloat32*)(vertex + cur->m_offset));
				break;
			}
			case DST_TYPE_FLOAT32_CLAMP1:
			{
				hkSimdReal p;
				p.load<1>(reals + cur->m_realOffset);
				p.mul(cur->m_recipWeight);
				p.setClampedZeroOne(p);
				p.store<1>((hkFloat32*)(vertex + cur->m_offset));
				break;
			}
			case DST_TYPE_FLOAT32_CLAMP2:
			{
				hkVector4 p;
				p.load<2,HK_IO_NATIVE_ALIGNED>(reals + cur->m_realOffset);
				p.mul(cur->m_recipWeight);
				p.setClamped(p,zeros,ones);
				p.store<2,HK_IO_NATIVE_ALIGNED>((hkFloat32*)(vertex + cur->m_offset));
				break;
			}
			case DST_TYPE_FLOAT32_CLAMP3:
			{
				hkVector4 p;
				p.load<3,HK_IO_NATIVE_ALIGNED>(reals + cur->m_realOffset);
				p.mul(cur->m_recipWeight);
				p.setClamped(p,zeros,ones);
				p.store<3,HK_IO_NATIVE_ALIGNED>((hkFloat32*)(vertex + cur->m_offset));
				break;
			}
			case DST_TYPE_FLOAT32_CLAMP4:
			{
				hkVector4 p;
				p.load<4,HK_IO_NATIVE_ALIGNED>(reals + cur->m_realOffset);
				p.mul(cur->m_recipWeight);
				p.setClamped(p,zeros,ones);
				p.store<4,HK_IO_NATIVE_ALIGNED>((hkFloat32*)(vertex + cur->m_offset));
				break;
			}
			case DST_TYPE_ARGB32:
			{
				const hkFloat32* srcV = (const hkFloat32*)(reals + cur->m_realOffset);
				const hkFloat32* endV = srcV + (cur->m_numValues * 4);
				hkUint32* dst = (hkUint32*)(vertex + cur->m_offset);

				for (; srcV < endV; srcV += 4, dst++)
				{
					hkVector4 v; v.load<4,HK_IO_NATIVE_ALIGNED>(srcV);
					v.mul(hkSimdReal_255 * cur->m_recipWeight);
					v.setClamped(v,zeros,all255);
					// Add the magic no, and extract the values
					v.add(magic);
#if defined(HK_REAL_IS_DOUBLE)
					HK_ASSERT2(0x256,false,"magic doesnt work for doubles");
#endif
					const hkUint32* iv = (const hkUint32*)&v(0);

					const hkUint32 packed =  (iv[0] & 0xff) | 
									  ((iv[1] & 0xff) << 8) |
									  ((iv[2] & 0xff) << 16) |
									  ((iv[3] & 0xff) << 24);
					*dst = packed;
				}
				break;
			}
			case DST_TYPE_FLOAT32_GENERAL:
			{
				const hkFloat32* srcV = (const hkFloat32*)(reals + cur->m_realOffset);
				const hkFloat32* endV = srcV + cur->m_numValues;
				hkFloat32* dst = (hkFloat32*)(vertex + cur->m_offset);
				const hkSimdReal recipWeight = cur->m_recipWeight;

				while (srcV < endV)
				{
					hkSimdReal s; 
					s.load<1>(srcV++);
					(s * recipWeight).store<1>(dst++);
				}
				break;
			}
			default:
			{
				// Can't handle this
				HK_ASSERT(0x32423423, !"Unhandled type");
			}
		}
	}
}

#ifdef HK_DEBUG

namespace // anonymous
{
struct Vertex
{
	hkFloat32 m_position[3];
	hkColor::Argb m_color;
	hkFloat32 m_texCoords[2];
};
}

/* static */void hkVertexFloat32Converter::selfTest()
{
	hkAabb aabb; 
	aabb.m_min.setAll(-2);
	aabb.m_max.setAll(2);

	hkVertexFloat32Converter converter;

	Vertex srcVertex[] = 
	{
		{
			{ 1, 2, -1 },
			0xffa02000,
			{0.5f, 0.25f }
		}
	};

	hkVertexFormat format;
	format.addElement(hkVertexFormat::USAGE_POSITION, hkVertexFormat::TYPE_FLOAT32, 3);
	format.addElement(hkVertexFormat::USAGE_COLOR, hkVertexFormat::TYPE_ARGB32, 1);
	format.addElement(hkVertexFormat::USAGE_TEX_COORD, hkVertexFormat::TYPE_FLOAT32, 2);
	format.makeCanonicalOrder();

	converter.init(format, aabb, true);

	hkFloat32 reals[9];
	converter.convertVertexToFloat32(&srcVertex, reals);

	Vertex dstVertex;
	converter.convertFloat32ToVertex(reals, (void*)&dstVertex);

	

}

#endif

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
