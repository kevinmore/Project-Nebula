/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Math/Vector/hkIntVector.h>
#include <Common/Base/Container/BitField/hkBitField.h>

/* static */hkResult hkMeshVertexBufferUtil::getElementVectorArray(const LockedVertices& lockedVertices, int bufferIndex, hkFloat32* out)
{
    HK_ASSERT(0x8270fabb, bufferIndex >= 0 && bufferIndex < lockedVertices.m_numBuffers);
    return getElementVectorArray(lockedVertices.m_buffers[bufferIndex], out, lockedVertices.m_numVertices);
}

hkResult HK_CALL hkMeshVertexBufferUtil::getElementVectorArray(const LockedVertices& lockedVertices, int bufferIndex, const hkBitField& verticesToRetrieve, hkFloat32* dst)
{
	HK_ASSERT(0x8270dabb, bufferIndex >= 0 && bufferIndex < lockedVertices.m_numBuffers);

	void* srcPtr = lockedVertices.m_buffers[bufferIndex].m_start;
	hkMeshVertexBuffer::LockedVertices::Buffer lBuffer = lockedVertices.m_buffers[bufferIndex];

	for (int vi = 0, k = 0; vi < verticesToRetrieve.getSize(); vi++)
	{
		if ( !verticesToRetrieve.get(vi) )
		{
			continue;
		}

		lBuffer.m_start = hkAddByteOffset(srcPtr, lBuffer.m_stride * vi);
		if ( getElementVectorArray(lBuffer, &dst[k << 2], 1) != HK_SUCCESS )
		{
			return HK_FAILURE;
		}
		k++;
	}

	return HK_SUCCESS;
}

/* static */hkResult hkMeshVertexBufferUtil::getIndexedElementVectorArray(const Buffer& buffer, const int* indices, hkFloat32* out, int numVertices)
{
    hkVertexFormat::Element element = buffer.m_element;

	// Force ARGB32 for color
	if ( (element.m_usage == hkVertexFormat::USAGE_COLOR) && (element.m_numValues == 4) && (element.m_dataType == hkVertexFormat::TYPE_UINT8) )
	{
		element.m_dataType = hkVertexFormat::TYPE_ARGB32;
	}

    switch (element.m_dataType)
    {
        case hkVertexFormat::TYPE_ARGB32:
        {
            for (int i = 0; i < numVertices; i++)
            {
				const hkUint8* cur = ((const hkUint8*)buffer.m_start) + buffer.m_stride * indices[i];

                hkInt32 packed = *(hkInt32*)cur;
                hkIntVector v; v.set(packed & 0xff, (packed >> 8)& 0xff, (packed >> 16)& 0xff, (packed >> 24)& 0xff);
				hkVector4 fv; v.convertS32ToF32(fv);
				fv.mul(hkSimdReal_Inv_255);
                fv.store<4,HK_IO_NATIVE_ALIGNED>(&out[i*4]);
            }
            break;
        }
        case hkVertexFormat::TYPE_FLOAT32:
        {
            int numValues = element.m_numValues;
            if (numValues > 4)
            {
                numValues = 4;
            }
			switch (numValues)
			{
				case 0: break;
				case 1:
				{
					for (int i = 0; i < numVertices; i++)
					{
						const hkFloat32* src = (const hkFloat32*)(((const hkUint8*)buffer.m_start) + buffer.m_stride * indices[i]);
						out[4*i]   = src[0];
						out[4*i+1] = 0.0f;
						out[4*i+2] = 0.0f;
						out[4*i+3] = 0.0f;
					}
					break;
				}
				case 2:
				{
					for (int i = 0; i < numVertices; i++)
					{
						const hkFloat32* src = (const hkFloat32*)(((const hkUint8*)buffer.m_start) + buffer.m_stride * indices[i]);
						out[4*i]   = src[0];
						out[4*i+1] = src[1];
						out[4*i+2] = 0.0f;
						out[4*i+3] = 0.0f;
					}
					break;
				}
				case 3:
				{
					for (int i = 0; i < numVertices; i++)
					{
						const hkFloat32* src = (const hkFloat32*)(((const hkUint8*)buffer.m_start) + buffer.m_stride * indices[i]);
						out[4*i]   = src[0];
						out[4*i+1] = src[1];
						out[4*i+2] = src[2];
						out[4*i+3] = 0.0f;
					}
					break;
				}
				case 4:
				{
					for (int i = 0; i < numVertices; i++)
					{
						const hkFloat32* src = (const hkFloat32*)(((const hkUint8*)buffer.m_start) + buffer.m_stride * indices[i]);
						out[4*i]   = src[0];
						out[4*i+1] = src[1];
						out[4*i+2] = src[2];
						out[4*i+3] = src[3];
					}
					break;
				}
			}
            break;
        }
        default:
        {
            HK_ASSERT(0xd8279a2c, !"Unable to convert type");
            return HK_FAILURE;
        }
    }
    return HK_SUCCESS;
}

/* static */hkResult hkMeshVertexBufferUtil::getElementVectorArray(const Buffer& buffer, hkFloat32* out, int numVertices)
{
    const hkVertexFormat::Element& element = buffer.m_element;

    switch (element.m_dataType)
    {
        case hkVertexFormat::TYPE_ARGB32:
        {
            /// Convert them all to single precision float
            const hkUint8* cur = (const hkUint8*)buffer.m_start;
            const int stride = buffer.m_stride;

            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkUint32 packed = *(hkUint32*)cur;
                hkIntVector v; v.set(packed & 0xff, (packed >> 8)& 0xff, (packed >> 16)& 0xff, (packed >> 24)& 0xff);
				hkVector4 fv; v.convertS32ToF32(fv);
				fv.mul(hkSimdReal_Inv_255);
                fv.store<4,HK_IO_NATIVE_ALIGNED>(&out[4*i]);
            }
            break;
        }
        case hkVertexFormat::TYPE_FLOAT32:
        {
            int numValues = element.m_numValues;
            if (numValues < 4)
            {
                hkString::memClear16(out, (4*sizeof(hkFloat32))/16 * numVertices);
            }
            if (numValues > 4)
            {
                numValues = 4;
            }
            stridedCopy(buffer.m_start, buffer.m_stride, out, 4*sizeof(hkFloat32), numValues * sizeof(hkFloat32), numVertices);
            break;
        }
        default:
        {
            HK_ASSERT(0xd8279a2c, !"Unable to convert type");
            return HK_FAILURE;
        }
    }

	// If we are requested to return bone weights with the 4-th implied, compute that now
	if ( element.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED )
	{
		for (int k = numVertices - 1; k >= 0; k--)
		{
			hkVector4 v;		v.load<3, HK_IO_NATIVE_ALIGNED>(&out[k << 2]);	// [x, y, z, *]
			hkSimdReal sumXYZ	= v.horizontalAdd<3>();							// (x + y + z)
			hkSimdReal w;		w.setSub(hkSimdReal_1, sumXYZ);					// 1 - (x + y + z)
								sumXYZ.setZero();
								w.setMax(w, sumXYZ);							// Do not allow negative weights
								v.setComponent<3>(w);
			v.store<4, HK_IO_NATIVE_ALIGNED>(&out[k << 2]);
		}
	}

    return HK_SUCCESS;
}

/* static */hkResult hkMeshVertexBufferUtil::setElementVectorArray(const LockedVertices& lockedVertices, int bufferIndex, const hkFloat32* src)
{
    HK_ASSERT(0x8270eabb, bufferIndex >= 0 && bufferIndex < lockedVertices.m_numBuffers);
    return setElementVectorArray(lockedVertices.m_buffers[bufferIndex], src, lockedVertices.m_numVertices);
}

hkResult HK_CALL hkMeshVertexBufferUtil::setElementVectorArray(const LockedVertices& srcLockedVertices, int bufferIndex, int dstStartVertex, const hkFloat32* src, int numVertices)
{
	// Alter the lockedVerts and fall-back to the original implementation
	hkMeshVertexBuffer::LockedVertices::Buffer buffer = srcLockedVertices.m_buffers[bufferIndex];
	buffer.m_start = hkAddByteOffset(buffer.m_start, buffer.m_stride * dstStartVertex);

	return setElementVectorArray(buffer, src, numVertices);
}

/* static */hkResult hkMeshVertexBufferUtil::setElementVectorArray(const Buffer& buffer, const hkFloat32* src, int numVertices)
{
    const hkVertexFormat::Element& element = buffer.m_element;

    switch (element.m_dataType)
    {
        case hkVertexFormat::TYPE_ARGB32:
        {
            const hkUint8* cur = (const hkUint8*)buffer.m_start;
            const int stride = buffer.m_stride;
            hkVector4 max = hkVector4::getConstant<HK_QUADREAL_255>();
            hkVector4 min; min.setZero();

            for (int i = 0; i < numVertices; i++, cur += stride)
            {
				hkVector4 source; source.load<4,HK_IO_NATIVE_ALIGNED>(&src[4*i]);
                hkVector4 v; v.setMul(max, source);
				v.setClamped(v, min, max);

				hkIntVector iv; iv.setConvertF32toS32(v);
                hkUint32 res = iv.getU32<0>() | (iv.getU32<1>() << 8) | (iv.getU32<2>() << 16) | (iv.getU32<3>() << 24);

                *(hkUint32*)cur = res;
            }
            break;
        }
        case hkVertexFormat::TYPE_FLOAT32:
        {
			int numValues = element.m_numValues;
            int elementSize = hkSizeOf(hkFloat32) * numValues;
            if (numValues > 4)
            {
                // Clear the remaining data
                stridedZero((char*)buffer.m_start + sizeof(hkFloat32) * 4, buffer.m_stride, elementSize - hkSizeOf(hkFloat32) * 4, numVertices);
                numValues = 4;
            }
            stridedCopy(src, 4*sizeof(hkFloat32), buffer.m_start, buffer.m_stride, elementSize, numVertices);
            break;
        }
        default:
        {
            HK_ASSERT(0xd8279a2b, !"Unable to convert type");
            return HK_FAILURE;
        }
    }
    return HK_SUCCESS;
}

/* static */hkResult hkMeshVertexBufferUtil::getElementIntArray(const LockedVertices& lockedVertices, int bufferIndex, int* dst)
{
    HK_ASSERT(0x8270dabb, bufferIndex >= 0 && bufferIndex < lockedVertices.m_numBuffers);
    const hkMeshVertexBuffer::LockedVertices::Buffer& buffer = lockedVertices.m_buffers[bufferIndex];
    const hkVertexFormat::Element& element = buffer.m_element;

    const int numVertices = lockedVertices.m_numVertices;
    int numValues = element.m_numValues;
    const hkUint8* cur = (const hkUint8*)buffer.m_start;
    const int stride = buffer.m_stride;

    switch (element.m_dataType)
    {
        case hkVertexFormat::TYPE_INT8:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                const hkInt8* src = (const hkInt8*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    *dst++ = int(src[j]);
                }
            }
            break;
        }
        case hkVertexFormat::TYPE_UINT8:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                const hkUint8* src = (const hkUint8*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    *dst++ = int(src[j]);
                }
            }
            break;

        }
		case hkVertexFormat::TYPE_INT16:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                const hkInt16* src = (const hkInt16*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    *dst++ = int(src[j]);
                }
            }
            break;
        }
        case hkVertexFormat::TYPE_UINT16:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                const hkUint16* src = (const hkUint16*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    *dst++ = int(src[j]);
                }
            }
            break;
        }
        case hkVertexFormat::TYPE_INT32:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                const hkInt32* src = (const hkInt32*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    *dst++ = int(src[j]);
                }
            }
            break;
        }
		case hkVertexFormat::TYPE_FLOAT32:
			{
				for (int i = 0; i < numVertices; i++, cur += stride)
				{
					const float* src = (const float*)cur;
					for (int j = 0; j < numValues; j++)
					{
						*dst++ = int(src[j]);
					}
				}
				break;
			}
        case hkVertexFormat::TYPE_UINT32:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                const hkUint32* src = (const hkUint32*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    *dst++ = int(src[j]);
                }
            }
            break;
        }
        case hkVertexFormat::TYPE_UINT8_DWORD:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                const hkUint32* src = (const hkUint32*)cur;
				HK_ASSERT(0x49d3af33, numValues == 4);

                const hkUint32 v = *src;
                *dst++ = v & 0xff;
                *dst++ = (v >> 8) & 0xff;
                *dst++ = (v >> 16) & 0xff;
                *dst++ = (v >> 24) & 0xff;
            }
            break;
        }

		case hkVertexFormat::TYPE_ARGB32:
			{
				for (int i = 0; i < numVertices; i++, cur += stride)
				{
					HK_ASSERT(0x49d3af33, numValues == 4);

#ifdef HK_PLATFORM_XBOX360
					*dst++ = cur[2] & 0xFF;
					*dst++ = cur[3] & 0xFF;
					*dst++ = cur[1] & 0xFF;
					*dst++ = cur[0] & 0xFF;
#else
					*dst++ = cur[0] & 0xff;
					*dst++ = cur[1] & 0xff;
					*dst++ = cur[2] & 0xff;
					*dst++ = cur[3] & 0xff;
#endif
				}
			}
			break;

        default:
        {
            HK_ASSERT(0xd8279a2a, !"Unable to convert type");
            return HK_FAILURE;
        }
    }

    return HK_SUCCESS;
}

hkResult HK_CALL hkMeshVertexBufferUtil::getElementIntArray(const LockedVertices& lockedVertices, int bufferIndex, const hkBitField& verticesToRetrieve, int* dst, int dstMaxSize)
{
	HK_ASSERT(0x8270dabb, bufferIndex >= 0 && bufferIndex < lockedVertices.m_numBuffers);

	LockedVertices lVerts	= lockedVertices;
	lVerts.m_numVertices	= 1;
	void* srcPtr			= lockedVertices.m_buffers[bufferIndex].m_start;
	hkMeshVertexBuffer::LockedVertices::Buffer& lBuffer = lVerts.m_buffers[bufferIndex];

	for (int vi = 0, k = 0; vi < verticesToRetrieve.getSize(); vi++)
	{
		if ( !verticesToRetrieve.get(vi) )
		{
			continue;
		}

		lBuffer.m_start = hkAddByteOffset(srcPtr, lBuffer.m_stride * vi);
		HK_ASSERT(0x510d31a8, (dstMaxSize < 0) || (k < dstMaxSize));
		if ( getElementIntArray(lVerts, bufferIndex, &dst[k]) != HK_SUCCESS )
		{
			return HK_FAILURE;
		}
		k += lBuffer.m_element.m_numValues;
	}

	return HK_SUCCESS;
}

/* static */hkResult hkMeshVertexBufferUtil::setElementIntArray(const LockedVertices& lockedVertices, int bufferIndex, const int* src)
{
    HK_ASSERT(0x8270cabb, bufferIndex >= 0 && bufferIndex < lockedVertices.m_numBuffers);
    const hkMeshVertexBuffer::LockedVertices::Buffer& buffer = lockedVertices.m_buffers[bufferIndex];
    const hkVertexFormat::Element& element = buffer.m_element;

    const int numVertices = lockedVertices.m_numVertices;
    int numValues = element.m_numValues;
    hkUint8* cur = (hkUint8*)buffer.m_start;
    const int stride = buffer.m_stride;

    switch (element.m_dataType)
    {
        case hkVertexFormat::TYPE_INT8:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkInt8* dst = (hkInt8*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    dst[j] = hkInt8(*src++);
                }
            }
            break;
        }
        case hkVertexFormat::TYPE_UINT8:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkUint8* dst = (hkUint8*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    dst[j]= hkUint8(*src++);
                }
            }
            break;

        }
		case hkVertexFormat::TYPE_INT16:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkInt16* dst = (hkInt16*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    dst[j]= hkInt16(*src++);
                }
            }
            break;
        }
        case hkVertexFormat::TYPE_UINT16:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkUint16* dst = (hkUint16*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    dst[j]= hkUint16(*src++);
                }
            }
            break;
        }
        case hkVertexFormat::TYPE_INT32:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkInt32* dst = (hkInt32*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    dst[j]= hkInt32(*src++);
                }
            }
            break;
        }
		case hkVertexFormat::TYPE_FLOAT32:
			{
				for (int i = 0; i < numVertices; i++, cur += stride)
				{
					float* dst = (float*)cur;
					for (int j = 0; j < numValues; j++)
					{
						dst[j]= float(*src++);
					}
				}
				break;
			}
        case hkVertexFormat::TYPE_UINT32:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkUint32* dst = (hkUint32*)cur;
                for (int j = 0; j < numValues; j++)
                {
                    dst[j] = hkUint32(*src++);
                }
            }
            break;
        }
        case hkVertexFormat::TYPE_UINT8_DWORD:
        {
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkUint32* dst = (hkUint32*)cur;
				HK_ASSERT(0x49d3af33, numValues == 4);

				hkUint32 v = (*src++) & 0xff;
				v = (v << 8) | ((*src++) & 0xff);
				v = (v << 8) | ((*src++) & 0xff);
				v = (v << 8) | ((*src++) & 0xff);

				*dst = v;
            }
            break;
        }

		case hkVertexFormat::TYPE_ARGB32:
			{
				 for (int i = 0; i < numVertices; i++, cur += stride)
				 {
#ifdef HK_PLATFORM_XBOX360
					 cur[0] = src[2] & 0xFF;
					 cur[1] = src[3] & 0xFF;
					 cur[2] = src[1] & 0xFF;
					 cur[3] = src[0] & 0xFF;	src += 4;
#else
					 cur[0] = src[0] & 0xFF;
					 cur[1] = src[1] & 0xFF;
					 cur[2] = src[2] & 0xFF;
					 cur[3] = src[3] & 0xFF;	src += 4;
#endif
				 }
			}
			break;

        default:
        {
            HK_ASSERT(0xd8279a29, !"Unable to convert type");
            return HK_FAILURE;
        }
    }
    return HK_SUCCESS;
}

hkResult hkMeshVertexBufferUtil::setElementIntArray(const LockedVertices& srcLockedVertices, int bufferIndex, int dstStartVertex, const int* src, int numVertices)
{
	// Alter the lockedVerts and fall-back to the original implementation
	hkMeshVertexBuffer::LockedVertices lockedVerts		= srcLockedVertices;
	hkMeshVertexBuffer::LockedVertices::Buffer& buffer	= lockedVerts.m_buffers[bufferIndex];

	lockedVerts.m_numVertices	= numVertices;
	buffer.m_start				= hkAddByteOffset(buffer.m_start, buffer.m_stride * dstStartVertex);

	return setElementIntArray(lockedVerts, bufferIndex, src);
}


/* static */void hkMeshVertexBufferUtil::stridedZero(void* dstIn, int dstStride, int elementSize, int numVertices)
{
    if (elementSize == 0)
    {
        return;
    }
    if (dstStride == elementSize)
    {
        hkString::memSet(dstIn, 0, elementSize * numVertices);
        return;
    }

    char* dst = (char*)dstIn;
    char* dstEnd = dst + (elementSize * numVertices);

    if ((elementSize & 3) == 0)
    {
        const int numWords = elementSize >> 2;
        switch (numWords)
        {
            case 1:
            {
                for (; dst != dstEnd; dst += dstStride)
                {
                    hkUint32* o = (hkUint32*)dst;
                    o[0] = 0;
                }
                return;
            }
            case 2:
            {
                for (; dst != dstEnd; dst += dstStride)
                {
                    hkUint32* o = (hkUint32*)dst;
                    o[0] = 0;
                    o[1] = 0;
                }
                return;
            }
            case 3:
            {
                for (; dst != dstEnd; dst += dstStride)
                {
                    hkUint32* o = (hkUint32*)dst;
                    o[0] = 0;
                    o[1] = 0;
                    o[2] = 0;
                }
                return;
            }
            case 4:
            {
                for (; dst != dstEnd; dst += dstStride)
                {
                    hkUint32* o = (hkUint32*)dst;
                    o[0] = 0;
                    o[1] = 0;
                    o[2] = 0;
                    o[3] = 0;
                }
                return;
            }
        }
    }

    // Okay - lets just do it the slow, but reliable way
    for (; dst != dstEnd; dst += dstStride)
    {
        hkString::memSet(dst, 0, elementSize);
    }
}

/* static */void hkMeshVertexBufferUtil::stridedCopy(const void* srcIn, int srcStride, void* dstIn, int dstStride, int elementSize, int numVertices)
{
    if (elementSize == 0)
    {
        return;
    }
    if (srcStride == dstStride && srcStride == elementSize)
    {
        hkString::memCpy(dstIn, srcIn, elementSize * numVertices);
        return;
    }

    const char* src = (const char*)srcIn;
    char* dst = (char*)dstIn;

    if ((elementSize & 3) == 0)
    {
        const char* srcEnd = src + srcStride * numVertices;

        const int numWords = elementSize >> 2;
        switch (numWords)
        {
            case 1:
            {
                for (; src != srcEnd; src += srcStride, dst += dstStride)
                {
                    hkUint32* o = (hkUint32*)dst;
                    const hkUint32* i = (const hkUint32*)src;
                    o[0] = i[0];
                }
                return;
            }
            case 2:
            {
                for (; src != srcEnd; src += srcStride, dst += dstStride)
                {
                    hkUint32* o = (hkUint32*)dst;
                    const hkUint32* i = (const hkUint32*)src;
                    o[0] = i[0];
                    o[1] = i[1];
                }
                return;
            }
            case 3:
            {
                for (; src != srcEnd; src += srcStride, dst += dstStride)
                {
                    hkUint32* o = (hkUint32*)dst;
                    const hkUint32* i = (const hkUint32*)src;
                    o[0] = i[0];
                    o[1] = i[1];
                    o[2] = i[2];
                }
                return;
            }
            case 4:
            {
                for (; src != srcEnd; src += srcStride, dst += dstStride)
                {
                    hkUint32* o = (hkUint32*)dst;
                    const hkUint32* i = (const hkUint32*)src;
                    o[0] = i[0];
                    o[1] = i[1];
                    o[2] = i[2];
                    o[3] = i[3];
                }
                return;
            }
        }
    }

    // Okay - lets just do it the slow, but reliable way
	const char* srcEnd = src + numVertices * srcStride;
    for (; src != srcEnd; src += srcStride, dst += dstStride)
    {
        hkString::memCpy(dst, src, elementSize);
    }
}

/* static */void hkMeshVertexBufferUtil::copy(const LockedVertices& srcVertices, const LockedVertices& dstVertices)
{
    const int numVertices = srcVertices.m_numVertices;
    const int numBuffers = srcVertices.m_numBuffers;
    if (dstVertices.m_numVertices != numVertices && numBuffers == dstVertices.m_numBuffers)
    {
        HK_ASSERT(0x42234324, "Cannot copy unless same number of vertice & format");
        return;
    }

    for (int i = 0; i < numBuffers; i++)
    {
        copy(srcVertices.m_buffers[i], dstVertices.m_buffers[i], numVertices);
    }
}

/* static */void hkMeshVertexBufferUtil::copy(const Buffer& srcBuffer, const Buffer& dstBuffer, int numVertices)
{
    HK_ASSERT(0x24234, srcBuffer.m_element.m_dataType == dstBuffer.m_element.m_dataType && srcBuffer.m_element.m_numValues == dstBuffer.m_element.m_numValues);

    int size = srcBuffer.m_element.calculateAlignedSize();

    // Now we need to do the copy
    stridedCopy(srcBuffer.m_start, srcBuffer.m_stride, dstBuffer.m_start, dstBuffer.m_stride, size, numVertices);
}

/* static */void hkMeshVertexBufferUtil::convert(const hkMeshVertexBuffer::LockedVertices& srcVertices, const hkMeshVertexBuffer::LockedVertices& dstVertices)
{
    const int numVertices = srcVertices.m_numVertices;
    const int numBuffers = srcVertices.m_numBuffers;
    if (dstVertices.m_numVertices != numVertices && numBuffers == dstVertices.m_numBuffers)
    {
        HK_ASSERT(0x42234324, "Cannot copy unless same number of vertice & format");
        return;
    }

    for (int i = 0; i < numBuffers; i++)
    {
        convert(srcVertices.m_buffers[i], dstVertices.m_buffers[i], numVertices);
    }
}

static void hkMeshVertexBufferUtil_convertUint8DwordToUint8(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
    const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	if (srcElement.m_numValues != dstElement.m_numValues ||
		( srcElement.m_dataType != hkVertexFormat::TYPE_UINT8_DWORD &&
		  srcElement.m_dataType != hkVertexFormat::TYPE_ARGB32 ) ||
		dstElement.m_dataType != hkVertexFormat::TYPE_UINT8)
	{
		HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
		return;
	}

    hkUint8* dstV = (hkUint8*)dstBuffer.m_start;
    const hkUint8* srcV = (const hkUint8*)srcBuffer.m_start;
    const int srcStride = srcBuffer.m_stride;
    const int dstStride = dstBuffer.m_stride;

    switch (srcElement.m_numValues)
    {
        case 1:
        {
            for (int i = 0; i < numVertices; i++)
            {
                hkUint32 v = *(const hkUint32*)srcV;
                dstV[0] = hkUint8(v);

                srcV += srcStride;
                dstV += dstStride;
            }
            return;
        }
        case 2:
        {
            for (int i = 0; i < numVertices; i++)
            {
                hkUint32 v = *(const hkUint32*)srcV;
                dstV[0] = hkUint8(v);
                dstV[1] = hkUint8(v >> 8);

                srcV += srcStride;
                dstV += dstStride;
            }
            return;
        }
        case 3:
        {
            for (int i = 0; i < numVertices; i++)
            {
                hkUint32 v = *(const hkUint32*)srcV;
                dstV[0] = hkUint8(v);
                dstV[1] = hkUint8(v >> 8);
                dstV[2] = hkUint8(v >> 16);

                srcV += srcStride;
                dstV += dstStride;
            }
            return;
        }
        case 4:
        {
            for (int i = 0; i < numVertices; i++)
            {
                hkUint32 v = *(const hkUint32*)srcV;
                dstV[0] = hkUint8(v);
                dstV[1] = hkUint8(v >> 8);
                dstV[2] = hkUint8(v >> 16);
                dstV[3] = hkUint8(v >> 24);

                srcV += srcStride;
                dstV += dstStride;
            }
            return;
        }
        default: break;
    }

    {
        int numDwords = srcElement.m_numValues / 4;
        int numBytes = srcElement.m_numValues & 3;

        for (int i = 0; i < numVertices; i++)
        {
            const hkUint32* src = (const hkUint32*)srcV;
            hkUint8* dst = (hkUint8*)dstV;

            for (int j = 0; j < numDwords; j++, src++, dst += 4)
            {
                hkUint32 v = *src;
                dst[0] = hkUint8(v);
                dst[1] = hkUint8(v >> 8);
                dst[2] = hkUint8(v >> 16);
                dst[3] = hkUint8(v >> 24);
            }

            switch (numBytes)
            {
                case 3:
                {
                    hkUint32 v = *src;
                    dst[0] = hkUint8(v);
                    dst[1] = hkUint8(v >> 8);
                    dst[2] = hkUint8(v >> 16);
                    break;
                }
                case 2:
                {
                    hkUint32 v = *src;
                    dst[0] = hkUint8(v);
                    dst[1] = hkUint8(v >> 8);
                    break;
                }
                case 1:
                {
                    hkUint32 v = *src;
                    dst[0] = hkUint8(v);
                    break;
                }
                case 0: break;
            }
        }
    }
}

static void hkMeshVertexBufferUtil_convertUint8DwordToInt16(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	if (srcElement.m_numValues != 4 ||
		dstElement.m_numValues != 4 ||
		srcElement.m_dataType != hkVertexFormat::TYPE_UINT8_DWORD ||
		dstElement.m_dataType != hkVertexFormat::TYPE_INT16)
	{
		HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
		return;
	}

	hkInt16* dstV = (hkInt16*)dstBuffer.m_start;
	const hkUint8* srcV = (const hkUint8*)srcBuffer.m_start;
	const int srcStride = srcBuffer.m_stride / sizeof(hkUint8);
	const int dstStride = dstBuffer.m_stride / sizeof(hkInt16);

	for (int i = 0; i < numVertices; i++)
	{
		hkUint32 v = *(const hkUint32*)srcV;
		dstV[0] = hkUint8(v);
		dstV[1] = hkUint8(v >> 8);
		dstV[2] = hkUint8(v >> 16);
		dstV[3] = hkUint8(v >> 24);

		srcV += srcStride;
		dstV += dstStride;
	}
}

static void hkMeshVertexBufferUtil_convertUint8DwordToArgb32(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	if (srcElement.m_numValues != dstElement.m_numValues ||
		srcElement.m_dataType != hkVertexFormat::TYPE_UINT8_DWORD ||
		dstElement.m_dataType != hkVertexFormat::TYPE_ARGB32)
	{
		HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
		return;
	}

	hkUint8* dstV = (hkUint8*)dstBuffer.m_start;
	const hkUint8* srcV = (const hkUint8*)srcBuffer.m_start;
	const int srcStride = srcBuffer.m_stride;
	const int dstStride = dstBuffer.m_stride;

	switch (srcElement.m_numValues)
	{
	case 4:
		{
			for (int i = 0; i < numVertices; i++)
			{
				dstV[0] = srcV[0];
				dstV[1] = srcV[1];
				dstV[2] = srcV[2];
				dstV[3] = srcV[3];

				srcV += srcStride;
				dstV += dstStride;
			}
			return;
		}	
	}

	HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
}

static void hkMeshVertexBufferUtil_convertArgb32ToToFloat32( const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices )
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	if ((srcElement.m_numValues != 1 && dstElement.m_numValues != 4) ||
		srcElement.m_dataType != hkVertexFormat::TYPE_ARGB32 ||
		dstElement.m_dataType != hkVertexFormat::TYPE_FLOAT32 )
	{
		HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
		return;
	}

	const hkUint32* src = (hkUint32*)srcBuffer.m_start;
	hkFloat32* dst = (hkFloat32*)dstBuffer.m_start;

	const int srcStride = srcBuffer.m_stride / sizeof(hkUint32);
	const int dstStride = dstBuffer.m_stride / sizeof(hkFloat32);

	for (int i = 0; i < numVertices; i++)
	{
		hkUint32 srcPacked = src[0];

		hkIntVector iv; iv.set((srcPacked >> 0) & 0xff, (srcPacked >> 8) & 0xff, (srcPacked >> 16)& 0xff, (srcPacked >> 24)& 0xff);

		hkVector4 v; 
		iv.convertS32ToF32(v); // signed conversion is ok because of & 0xff
		v.mul(hkSimdReal_Inv_255);
		v.store<4,HK_IO_NATIVE_ALIGNED>(dst);

		src += srcStride; // asserted above
		dst += dstStride;
	}
}

static void hkMeshVertexBufferUtil_convertUint8DwordToFloat32(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	if (srcElement.m_numValues != dstElement.m_numValues ||
		srcElement.m_dataType != hkVertexFormat::TYPE_UINT8_DWORD ||
		dstElement.m_dataType != hkVertexFormat::TYPE_FLOAT32)
	{
		HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
		return;
	}

	hkFloat32* dstV = (hkFloat32*)dstBuffer.m_start;
	const hkUint8* srcV = (const hkUint8*)srcBuffer.m_start;
	const int srcStride = srcBuffer.m_stride;
	const int dstStride = dstBuffer.m_stride / sizeof(hkFloat32);

	switch (srcElement.m_numValues)
	{
	case 4:
		{
			for (int i = 0; i < numVertices; i++)
			{
				hkIntVector iv; iv.set(srcV[0],srcV[1],srcV[2],srcV[3]);
				hkVector4 v; iv.convertS32ToF32(v);
				v.store<4,HK_IO_NATIVE_ALIGNED>(dstV);

				srcV += srcStride;
				dstV += dstStride;
			}
			return;
		}	
	}

	HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
}


static void hkMeshVertexBufferUtil_convertUint8ToUint8Dword(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
    const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
    const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	if (srcElement.m_numValues != dstElement.m_numValues ||
		srcElement.m_dataType != hkVertexFormat::TYPE_UINT8 ||
		( dstElement.m_dataType != hkVertexFormat::TYPE_UINT8_DWORD &&
		dstElement.m_dataType != hkVertexFormat::TYPE_ARGB32 ) )
	{
		HK_ASSERT(0xbd838dd8, !"Cannot do conversion");
		return;
	}

    hkUint8* dstV = (hkUint8*)dstBuffer.m_start;
    const hkUint8* srcV = (const hkUint8*)srcBuffer.m_start;
    const int srcStride = srcBuffer.m_stride;
    const int dstStride = dstBuffer.m_stride;

    switch (srcElement.m_numValues)
    {
        case 1:
        {
            for (int i = 0; i < numVertices; i++)
            {
                *(hkUint32*)dstV = hkUint32(srcV[0]);
                srcV += srcStride;
                dstV += dstStride;
            }
            return;
        }
        case 2:
        {
            for (int i = 0; i < numVertices; i++)
            {
                *(hkUint32*)dstV = hkUint32(srcV[0]) | (hkUint32(srcV[1]) << 8);
                srcV += srcStride;
                dstV += dstStride;
            }
            return;
        }
        case 3:
        {
            for (int i = 0; i < numVertices; i++)
            {
                *(hkUint32*)dstV = hkUint32(srcV[0]) | (hkUint32(srcV[1]) << 8) | (hkUint32(srcV[2]) << 16);
                srcV += srcStride;
                dstV += dstStride;
            }
            return;
        }
        case 4:
        {
            for (int i = 0; i < numVertices; i++)
            {
                *(hkUint32*)dstV = hkUint32(srcV[0]) | (hkUint32(srcV[1]) << 8) | (hkUint32(srcV[2]) << 16) | (hkUint32(srcV[3]) << 24);
                srcV += srcStride;
                dstV += dstStride;
            }
            return;
        }
        default: break;
    }

    {
        int numDwords = srcElement.m_numValues / 4;
        int numBytes = srcElement.m_numValues & 3;

        for (int i = 0; i < numVertices; i++)
        {
            const hkUint8* src = (const hkUint8*)srcV;
            hkUint32* dst = (hkUint32*)dstV;

            for (int j = 0; j < numDwords; j++, src += 4, dst ++)
            {
                *dst = hkUint32(src[0]) | (hkUint32(src[1]) << 8) | (hkUint32(src[2]) << 16) | (hkUint32(src[3]) << 24);
            }

            switch (numBytes)
            {
                case 3:
                {
                    *dst = hkUint32(src[0]) | (hkUint32(src[1]) << 8) | (hkUint32(src[2]) << 16);
                    break;
                }
                case 2:
                {
                    *dst = hkUint32(src[0]) | (hkUint32(src[1]) << 8);
                    break;
                }
                case 1:
                {
                    *dst = hkUint32(src[0]);
                    break;
                }
                default:
                case 0: break;
            }
        }
    }
}

static void hkMeshVertexBufferUtil_convertUint8ToFloat32(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	const int srcStride = srcBuffer.m_stride;
	const int dstStride = dstBuffer.m_stride;

	const hkUint8* srcV = (hkUint8*)srcBuffer.m_start;
	hkUint8* dstV = (hkUint8*)dstBuffer.m_start;

	if (srcElement.m_numValues == dstElement.m_numValues)
	{
		const int numValues = srcElement.m_numValues;
		for (int i = 0; i < numVertices; i++, dstV += dstStride, srcV += srcStride)
		{
			hkFloat32* dst = (hkFloat32*)dstV;
			for (int j = 0; j < numValues; j++)
			{
				hkSimdReal v; 
				v.setFromUint8(srcV[j]);
				v.mul(hkSimdReal_Inv_255);
				v.store<1>(&dst[j]);
			}
		}
		return;
	}

	if (srcElement.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS && dstElement.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED)
	{
		if (srcElement.m_numValues != dstElement.m_numValues - 1)
		{
			HK_ASSERT(0x324342, !"Wrong amount of values");
			return;
		}
		const int numValues = dstElement.m_numValues;
		for (int i = 0; i < numVertices; i++, dstV += dstStride, srcV += srcStride)
		{
			hkFloat32* dst = (hkFloat32*)dstV;
			for (int j = 0; j < numValues; j++)
			{
				hkSimdReal v; 
				v.setFromUint8(srcV[j]);
				v.mul(hkSimdReal_Inv_255);
				v.store<1>(&dst[j]);
			}
		}
		return;
	}

	if ( srcElement.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED || dstElement.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS)
	{
		if (dstElement.m_numValues != srcElement.m_numValues - 1)
		{
			HK_ASSERT(0x324342, !"Wrong amount of values");
			return;
		}
		const int numValues = dstElement.m_numValues;
		for (int i = 0; i < numVertices; i++, dstV += dstStride, srcV += srcStride)
		{
			hkSimdReal sum; sum.setZero();
			hkFloat32* dst = (hkFloat32*)dstV;
			for (int j = 0; j < numValues; j++)
			{
				hkSimdReal v; 
				v.setFromUint8(srcV[j]);
				sum.add(v);
				v.mul(hkSimdReal_Inv_255);
				v.store<1>(&dst[j]);
			}
			const hkSimdReal finalVal = (hkSimdReal_255 - sum) * hkSimdReal_Inv_255;
			finalVal.store<1>(&dst[numValues]);
		}
		return;
	}

	HK_ASSERT(0x8d7292c0, !"Don't know how to convert");
}

static void hkMeshVertexBufferUtil_convertUint8ToInt16(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	const int srcStride = srcBuffer.m_stride;
	const int dstStride = dstBuffer.m_stride;

	const hkUint8* srcV = (hkUint8*)srcBuffer.m_start;
	hkUint8* dstV = (hkUint8*)dstBuffer.m_start;

	if (srcElement.m_numValues == dstElement.m_numValues)
	{
		const int numValues = srcElement.m_numValues;
		for (int i = 0; i < numVertices; i++, dstV += dstStride, srcV += srcStride)
		{
			hkInt16* dst = (hkInt16*)dstV;
			for (int j = 0; j < numValues; j++)
			{
				dst[j] = srcV[j];
			}
		}
		return;
	}

	HK_ASSERT(0x8d7292c0, !"Don't know how to convert");
}

static void hkMeshVertexBufferUtil_convertUint8ToUint16(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	const int srcStride = srcBuffer.m_stride;
	const int dstStride = dstBuffer.m_stride;

	const hkUint8* srcV = (hkUint8*)srcBuffer.m_start;
	hkUint8* dstV = (hkUint8*)dstBuffer.m_start;

	if (srcElement.m_numValues == dstElement.m_numValues)
	{
		const int numValues = srcElement.m_numValues;
		for (int i = 0; i < numVertices; i++, dstV += dstStride, srcV += srcStride)
		{
			hkUint16* dst = (hkUint16*)dstV;
			for (int j = 0; j < numValues; j++)
			{
				dst[j] = srcV[j];
			}
		}
		return;
	}

	HK_ASSERT(0x8d7292c0, !"Don't know how to convert");
}

static void hkMeshVertexBufferUtil_convertFloat32ToUint8(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
    const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
    const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

    const int srcStride = srcBuffer.m_stride;
    const int dstStride = dstBuffer.m_stride;

    const hkUint8* srcV = (hkUint8*)srcBuffer.m_start;
    hkUint8* dstV = (hkUint8*)dstBuffer.m_start;

    if (srcElement.m_numValues == dstElement.m_numValues)
    {
        const int numValues = srcElement.m_numValues;
        for (int i = 0; i < numVertices; i++, dstV += dstStride, srcV += srcStride)
        {
            const hkFloat32* src = (const hkFloat32*)srcV;
            for (int j = 0; j < numValues; j++)
            {
				hkSimdReal v; v.load<1>(&src[j]);
				v.mul(hkSimdReal_255);
				hkInt32 dstI32;
				v.storeSaturateInt32(&dstI32);
                dstV[j] = hkUint8(dstI32);
            }
        }
        return;
    }

    if (srcElement.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS && dstElement.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED)
    {
        if (srcElement.m_numValues != dstElement.m_numValues - 1)
        {
            HK_ASSERT(0x324342, !"Wrong amount of values");
            return;
        }
        const int numValues = dstElement.m_numValues;
        for (int i = 0; i < numVertices; i++, dstV += dstStride, srcV += srcStride)
        {
            const hkFloat32* src = (const hkFloat32*)srcV;
            for (int j = 0; j < numValues; j++)
            {
				hkSimdReal v; v.load<1>(&src[j]);
				v.mul(hkSimdReal_255);
				hkInt32 dstI32;
				v.storeSaturateInt32(&dstI32);
				dstV[j] = hkUint8(dstI32);
            }
        }
        return;
    }

    if ( srcElement.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED || dstElement.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS)
    {
        if (dstElement.m_numValues != srcElement.m_numValues - 1)
        {
            HK_ASSERT(0x324342, !"Wrong amount of values");
            return;
        }
        const int numValues = dstElement.m_numValues;
        for (int i = 0; i < numVertices; i++, dstV += dstStride, srcV += srcStride)
        {
            hkSimdReal sum; sum.setZero();
            const hkFloat32* src = (const hkFloat32*)srcV;
			hkInt32 dstI32;
            for (int j = 0; j < numValues; j++)
            {
				hkSimdReal v; v.load<1>(&src[j]);
				v.mul(hkSimdReal_255);
				sum.add(v);
				v.storeSaturateInt32(&dstI32);
				dstV[j] = hkUint8(dstI32);
            }
			(hkSimdReal_255 - sum).storeSaturateInt32(&dstI32);
            dstV[numValues] = hkUint8(dstI32);
        }
        return;
    }

    HK_ASSERT(0x8d7292bf, !"Don't know how to convert");
}

static void hkMeshVertexBufferUtil_convertInt16ToUint8(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	const int srcStride = srcBuffer.m_stride;
	const int dstStride = dstBuffer.m_stride;

	const hkUint8* srcV = (hkUint8*)srcBuffer.m_start;
	hkUint8* dstV = (hkUint8*)dstBuffer.m_start;

	if (srcElement.m_numValues == dstElement.m_numValues)
	{
		const int numValues = srcElement.m_numValues;
		for (int i = 0; i < numVertices; i++, dstV += dstStride, srcV += srcStride)
		{
			const hkInt16* src = (const hkInt16*)srcV;
			for (int j = 0; j < numValues; j++)
			{
				dstV[j] = (hkUint8)src[j];
			}
		}
		return;
	}

	HK_ASSERT(0x8d7292bf, !"Don't know how to convert");
}

static void hkMeshVertexBufferUtil_convertFloat323FloatsToFloat324Floats(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	if (srcElement.m_numValues != 3 ||
		dstElement.m_numValues != 4 ||
		srcElement.m_dataType != hkVertexFormat::TYPE_FLOAT32 ||
		dstElement.m_dataType != hkVertexFormat::TYPE_FLOAT32 )
	{
		HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
		return;
	}

	hkFloat32* dstV = (hkFloat32*)dstBuffer.m_start;
	const hkFloat32* srcV = (const hkFloat32*)srcBuffer.m_start;
	
	const int srcStride = srcBuffer.m_stride / sizeof(hkFloat32);
	const int dstStride = dstBuffer.m_stride / sizeof(hkFloat32);

	for (int i = 0; i < numVertices; i++)
	{		
		dstV[0] = srcV[0];
		dstV[1] = srcV[1];
		dstV[2] = srcV[2];
		dstV[3] = 0.0f;

		srcV += srcStride; // asserted above
		dstV += dstStride;
	}	
}

static void hkMeshVertexBufferUtil_convertFloat32ToVector4(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	if (srcElement.m_numValues != 3 ||
		dstElement.m_numValues != 1 ||
		srcElement.m_dataType != hkVertexFormat::TYPE_FLOAT32 ||
		dstElement.m_dataType != hkVertexFormat::TYPE_VECTOR4 )
	{
		HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
		return;
	}

	hkFloat32* dstV = (hkFloat32*)dstBuffer.m_start;
	const hkFloat32* srcV = (const hkFloat32*)srcBuffer.m_start;

	const int srcStride = srcBuffer.m_stride / sizeof(hkFloat32);
	const int dstStride = dstBuffer.m_stride / sizeof(hkFloat32);

	for (int i = 0; i < numVertices; i++)
	{		
		dstV[0] = srcV[0];
		dstV[1] = srcV[1];
		dstV[2] = srcV[2];
		dstV[3] = 0;

		srcV += srcStride; // asserted above
		dstV += dstStride;
	}	
}

static void hkMeshVertexBufferUtil_convertVector4ToFloat32Floats(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numDstValues, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	if (srcElement.m_numValues != 1 ||
		dstElement.m_numValues != numDstValues ||
		srcElement.m_dataType != hkVertexFormat::TYPE_VECTOR4 ||
		dstElement.m_dataType != hkVertexFormat::TYPE_FLOAT32 )
	{
		HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
		return;
	}

	hkFloat32* dstV = (hkFloat32*)dstBuffer.m_start;
	const hkFloat32* srcV = (const hkFloat32*)srcBuffer.m_start;

	const int srcStride = srcBuffer.m_stride / sizeof(hkFloat32);
	const int dstStride = dstBuffer.m_stride / sizeof(hkFloat32);

	for (int i = 0; i < numVertices; i++)
	{
		for (int n=0; n<numDstValues; ++n)
			dstV[n] = srcV[n];
		
		srcV += srcStride; // asserted above
		dstV += dstStride;
	}	
}

static void hkMeshVertexBufferUtil_convertFloat32ToArgb32(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer, int numVertices)
{
	const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
	const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

	if (srcElement.m_numValues != 4 ||
		dstElement.m_numValues != 1 ||
		srcElement.m_dataType != hkVertexFormat::TYPE_FLOAT32 ||
		dstElement.m_dataType != hkVertexFormat::TYPE_ARGB32 )
	{
		HK_ASSERT(0xbd838dd9, !"Cannot do conversion");
		return;
	}

	hkUint32* dstV = (hkUint32*)dstBuffer.m_start;
	const hkFloat32* srcV = (const hkFloat32*)srcBuffer.m_start;
	const int srcStride = srcBuffer.m_stride / sizeof(hkFloat32);
	const int dstStride = dstBuffer.m_stride / sizeof(hkUint32);

	hkVector4 zeros; zeros.setZero();
	hkVector4 magic; magic.setAll(0x800000);
	hkVector4 all255 = hkVector4::getConstant<HK_QUADREAL_255>();

	for (int i = 0; i < numVertices; i++)
	{
		hkVector4 v; 

		v.load<4,HK_IO_NATIVE_ALIGNED>(srcV);		
		v.mul(all255);
		v.setMax(zeros, v);
		v.setMin(all255, v);
		v.add(magic);
#if defined(HK_REAL_IS_DOUBLE)
		HK_ASSERT2(0x344,false,"error magic bitpattern not suited for doubles");
#endif

		hkIntVector iv;
		iv.loadAsFloat32BitRepresentation(v);

		const hkUint32 packed =  (iv.getU32<0>() & 0xff) | ((iv.getU32<1>() & 0xff) << 8) | ((iv.getU32<2>() & 0xff) << 16) | ((iv.getU32<3>() & 0xff) << 24);
		*dstV = packed;

		srcV += srcStride;
		dstV += dstStride;
	}	
}

/* static */void hkMeshVertexBufferUtil::convert(const Buffer& srcBuffer, const Buffer& dstBuffer, int numVertices)
{
    const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
    const hkVertexFormat::Element& dstElement = dstBuffer.m_element;

    if (srcElement.m_dataType == dstElement.m_dataType && srcElement.m_numValues == dstElement.m_numValues)
    {
        // If they are the same format, we can just copy
        int size = srcElement.calculateAlignedSize();
        // Now we need to do the copy
        stridedCopy(srcBuffer.m_start, srcBuffer.m_stride, dstBuffer.m_start, dstBuffer.m_stride, size, numVertices);
        return;
    }

	if (   (srcElement.m_dataType == hkVertexFormat::TYPE_INT16 || srcElement.m_dataType == hkVertexFormat::TYPE_UINT16 )
		&& (dstElement.m_dataType == hkVertexFormat::TYPE_INT16 || dstElement.m_dataType == hkVertexFormat::TYPE_UINT16 )
		&&  dstElement.m_numValues == dstElement.m_numValues)
	{
		// If they are the same format, we can just copy
		int size = srcElement.calculateAlignedSize();
		// Now we need to do the copy
		stridedCopy(srcBuffer.m_start, srcBuffer.m_stride, dstBuffer.m_start, dstBuffer.m_stride, size, numVertices);
		return;
	}

	if (srcElement.m_dataType == dstElement.m_dataType && srcElement.m_numValues >= dstElement.m_numValues)
	{
		// If they are the same type, but less values, we can do a copy
		int size = dstElement.calculateAlignedSize();
		// Now we need to do the copy
		stridedCopy(srcBuffer.m_start, srcBuffer.m_stride, dstBuffer.m_start, dstBuffer.m_stride, size, numVertices);
		return;
	}

    if (srcElement.m_dataType == hkVertexFormat::TYPE_UINT8_DWORD)
    {
        switch (dstElement.m_dataType)
        {
            case hkVertexFormat::TYPE_UINT8:
            {
                hkMeshVertexBufferUtil_convertUint8DwordToUint8(srcBuffer, dstBuffer, numVertices);
                return;
            }
			case hkVertexFormat::TYPE_INT16:
			{
				hkMeshVertexBufferUtil_convertUint8DwordToInt16(srcBuffer, dstBuffer, numVertices);
				return;
			}
			case hkVertexFormat::TYPE_ARGB32:
			{
				hkMeshVertexBufferUtil_convertUint8DwordToArgb32(srcBuffer, dstBuffer, numVertices);
				return;
			}
			case hkVertexFormat::TYPE_FLOAT32:
			{
				hkMeshVertexBufferUtil_convertUint8DwordToFloat32(srcBuffer, dstBuffer, numVertices);
				return;
			}
		
            default: break;
        }
        HK_ASSERT(0x24234, !"Don't know how to convert");
        return;
    }

    if (srcElement.m_dataType == hkVertexFormat::TYPE_UINT8)
    {
        switch (dstElement.m_dataType)
        {
            case hkVertexFormat::TYPE_UINT8_DWORD:
			case hkVertexFormat::TYPE_ARGB32:
            {
                hkMeshVertexBufferUtil_convertUint8ToUint8Dword(srcBuffer, dstBuffer, numVertices);
                return;
            }
			case hkVertexFormat::TYPE_INT16:
			{
				hkMeshVertexBufferUtil_convertUint8ToInt16(srcBuffer, dstBuffer, numVertices);
				return;
			}
			case hkVertexFormat::TYPE_UINT16:
				{
					hkMeshVertexBufferUtil_convertUint8ToUint16(srcBuffer, dstBuffer, numVertices);
					return;
				}
            case hkVertexFormat::TYPE_FLOAT32:
            {
                hkMeshVertexBufferUtil_convertUint8ToFloat32(srcBuffer, dstBuffer, numVertices);
                return;
            }
            default: break;
        }

        HK_ASSERT(0x24234, !"Don't know how to convert");
        return;
    }

	if ( (srcElement.m_dataType == hkVertexFormat::TYPE_INT16) || (srcElement.m_dataType == hkVertexFormat::TYPE_UINT16) )
	{
		switch ( dstElement.m_dataType )
		{
		case hkVertexFormat::TYPE_UINT8:
			hkMeshVertexBufferUtil_convertInt16ToUint8(srcBuffer, dstBuffer, numVertices);
			return;

		default:
			break;
		}

		HK_ASSERT(0x24234, !"Don't know how to convert");
		return;
	}

    if (srcElement.m_dataType == hkVertexFormat::TYPE_FLOAT32)
    {
        switch (dstElement.m_dataType)
        {
            case hkVertexFormat::TYPE_UINT8:
            {
                hkMeshVertexBufferUtil_convertFloat32ToUint8(srcBuffer, dstBuffer, numVertices);
                return;
            }
            case hkVertexFormat::TYPE_FLOAT32:
            {
				hkMeshVertexBufferUtil_convertFloat323FloatsToFloat324Floats(srcBuffer, dstBuffer, numVertices);
				return;
            }
			case hkVertexFormat::TYPE_ARGB32:
			{
				hkMeshVertexBufferUtil_convertFloat32ToArgb32(srcBuffer, dstBuffer, numVertices);
				return;
			}
			case hkVertexFormat::TYPE_VECTOR4:
			{
				hkMeshVertexBufferUtil_convertFloat32ToVector4(srcBuffer, dstBuffer, numVertices);
				return;
			}
            default: break;
        }
        HK_ASSERT(0x24234, !"Don't know how to convert");
        return;
    }

	if (srcElement.m_dataType == hkVertexFormat::TYPE_VECTOR4)
	{
		switch (dstElement.m_dataType)
		{
			case hkVertexFormat::TYPE_FLOAT32:
			{
				if (srcElement.m_numValues == 1)
				{
					hkMeshVertexBufferUtil_convertVector4ToFloat32Floats(srcBuffer, dstBuffer, dstElement.m_numValues, numVertices);
					return;
				}
			}
			default: break;
		}
	}
	if (srcElement.m_dataType == hkVertexFormat::TYPE_UINT32)
	{
		if ( srcElement.m_numValues == 1 && dstElement.m_dataType == hkVertexFormat::TYPE_ARGB32 )
		{
			stridedCopy(srcBuffer.m_start, srcBuffer.m_stride, dstBuffer.m_start, dstBuffer.m_stride, sizeof(hkUint32) * 1 /*dstElement.m_numValues*/, numVertices);
			return;
		}

		if ( (srcElement.m_numValues == 1) && (srcElement.m_usage == hkVertexFormat::USAGE_COLOR) && (dstElement.m_dataType == hkVertexFormat::TYPE_FLOAT32) )
		{
			// Convert hkUint32 color. Assume its ARGB32, we don't know the format anyway!
			hkMeshVertexBuffer::LockedVertices::Buffer tempSrcBuffer = srcBuffer;
			tempSrcBuffer.m_element.m_dataType = hkVertexFormat::TYPE_ARGB32;
			hkMeshVertexBufferUtil_convertArgb32ToToFloat32(tempSrcBuffer, dstBuffer, numVertices);					
			return;
		}
	}
	if (srcElement.m_dataType == hkVertexFormat::TYPE_ARGB32)
	{
		if ( srcElement.m_numValues == 1 && dstElement.m_dataType == hkVertexFormat::TYPE_UINT32 )
		{
			stridedCopy(srcBuffer.m_start, srcBuffer.m_stride, dstBuffer.m_start, dstBuffer.m_stride, sizeof(hkUint32) * 1 /*dstElement.m_numValues*/, numVertices);
			return;
		}
		else if (srcElement.m_numValues == 1 && dstElement.m_dataType == hkVertexFormat::TYPE_FLOAT32 )
		{			
			hkMeshVertexBufferUtil_convertArgb32ToToFloat32(srcBuffer, dstBuffer, numVertices);					
			return;
		}
		else if (srcElement.m_numValues == 4 && dstElement.m_dataType == hkVertexFormat::TYPE_UINT8 )
		{			
			hkMeshVertexBufferUtil_convertUint8DwordToUint8(srcBuffer, dstBuffer, numVertices);					
			return;
		}
	}
	HK_WARN_ALWAYS( 0xabba4523, "Cannot convert vertex format from " << srcElement.m_dataType << " to " << dstElement.m_dataType);
    HK_ASSERT(0x24234, !"Don't know how to convert");
}

HK_FORCE_INLINE bool hkMeshVertexBufferUtil_orderBuffers(const hkMeshVertexBuffer::LockedVertices::Buffer& a, const hkMeshVertexBuffer::LockedVertices::Buffer& b)
{
    return a.m_start < b.m_start;
}

/* static */hkBool hkMeshVertexBufferUtil::isContiguous(const LockedVertices& lockedVertices, void** startOut, int& dataSize)
{
    typedef hkMeshVertexBuffer::LockedVertices::Buffer Buffer;

    const int numBuffers = lockedVertices.m_numBuffers;
    if (numBuffers <= 0)
    {
        // Not really contiguous as has no data
        return false;
    }

    if (numBuffers == 1)
    {
        *startOut = lockedVertices.m_buffers[0].m_start;
        dataSize = lockedVertices.m_buffers[0].m_element.calculateAlignedSize();

        return true;
    }

    // Reorder
    hkLocalArray<Buffer> buffers(numBuffers);
    buffers.setSizeUnchecked(numBuffers);

    hkString::memCpy(buffers.begin(), lockedVertices.m_buffers, numBuffers * sizeof(Buffer));

    hkSort(buffers.begin(), numBuffers, hkMeshVertexBufferUtil_orderBuffers);

    hkUint8* start = (hkUint8*)buffers[0].m_start;
    hkUint8* cur = start;
    for (int i = 0; i < numBuffers; i++)
    {
        const Buffer& buffer = lockedVertices.m_buffers[i];
        if (buffer.m_start != (void*)cur)
        {
            return false;
        }
        // Next
        int size = buffer.m_element.calculateAlignedSize();
        cur += size;
    }

    dataSize = int(cur - start);
    *startOut = start;

    return true;
}

/* static */void hkMeshVertexBufferUtil::partitionVertexFormat(const hkVertexFormat& format, hkVertexFormat& sharedFormat, hkVertexFormat& instanceFormat)
{
    sharedFormat.clear();
    instanceFormat.clear();

    {
        const int numElements = format.m_numElements;
        for (int i = 0; i < numElements; i++)
        {
			const hkVertexFormat::Element& ele = format.m_elements[i];
			if (ele.m_flags.anyIsSet(hkVertexFormat::FLAG_NOT_SHARED))
            {
                instanceFormat.addElement(ele);
            }
            else
            {
                sharedFormat.addElement(ele);
            }
        }
    }
}

/* static */void HK_CALL hkMeshVertexBufferUtil::convert(hkMeshVertexBuffer* src, hkMeshVertexBuffer* dst)
{
	hkMeshVertexBuffer::LockedVertices srcLocked;
	hkMeshVertexBuffer::LockedVertices dstLocked;
	hkMeshVertexBuffer::LockInput lockInput;

	if (src->getNumVertices() != dst->getNumVertices())
	{
		HK_ASSERT(0x277abd32, !"Buffers have different amount of vertices");
		return;
	}

	hkMeshVertexBuffer::LockResult lockRes = src->lock(lockInput, srcLocked);
	if (lockRes != hkMeshVertexBuffer::RESULT_SUCCESS)
	{
		HK_ASSERT(0x277abd31, !"Could not lock");
		return;
	}

	lockRes = dst->lock(lockInput, dstLocked);
	if (lockRes != hkMeshVertexBuffer::RESULT_SUCCESS)
	{
		src->unlock(srcLocked);
		HK_ASSERT(0x277abd30, !"Could not lock");
		return;
	}

	hkVertexFormat srcFormat, dstFormat;
	src->getVertexFormat(srcFormat);
	dst->getVertexFormat(dstFormat);

	if (srcFormat == dstFormat)
	{
		copy(srcLocked, dstLocked);
	}
	else
	{
		for (int i = 0; i < srcFormat.m_numElements; i++)
		{
			const hkVertexFormat::Element& ele = srcFormat.m_elements[i];
			int index = dstFormat.findElementIndex(ele.m_usage, ele.m_subUsage);

			// Test for weights / last implied
			if ( (index < 0) && ((ele.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS) || (ele.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED)) )
			{
				// If we tried one type of weights and failed, try the other one as well
				const hkVertexFormat::ComponentUsage otherWeightsUsage = (ele.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS) ? hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED : hkVertexFormat::USAGE_BLEND_WEIGHTS;
				index = dstFormat.findElementIndex(otherWeightsUsage, ele.m_subUsage);
			}

			if (index >= 0)
			{
				convert(srcLocked.m_buffers[i], dstLocked.m_buffers[index], srcLocked.m_numVertices);
			}
		}
	}

	src->unlock(srcLocked);
	dst->unlock(dstLocked);
}


/* static */hkResult hkMeshVertexBufferUtil::getElementVectorArray(hkMeshVertexBuffer* vertexBuffer, hkVertexFormat::ComponentUsage usage, int subUsage, hkArray<hkVector4>& vectorsOut)
{
    hkVertexFormat vertexFormat;
    vertexBuffer->getVertexFormat(vertexFormat);

    int elementIndex = vertexFormat.findElementIndex(usage, subUsage);

    if (elementIndex < 0)
    {
        return HK_FAILURE;
    }

    hkMeshVertexBuffer::LockInput lockInput;
    hkMeshVertexBuffer::PartialLockInput partialLockInput;

    partialLockInput.m_numLockFlags = 1;
    partialLockInput.m_elementIndices[0] = elementIndex;
    partialLockInput.m_lockFlags[0] = hkMeshVertexBuffer::ACCESS_READ | hkMeshVertexBuffer::ACCESS_ELEMENT_ARRAY;

    hkMeshVertexBuffer::LockedVertices lockedVertices;

    hkMeshVertexBuffer::LockResult result = vertexBuffer->partialLock(lockInput, partialLockInput, lockedVertices);
    if (result != hkMeshVertexBuffer::RESULT_SUCCESS)
    {
        return HK_FAILURE;
    }

    vectorsOut.setSize(lockedVertices.m_numVertices);
	hkArray<hkFloat32>::Temp va; va.setSize(4*lockedVertices.m_numVertices);
    vertexBuffer->getElementVectorArray(lockedVertices, 0, va.begin());
	for (int i=0; i<lockedVertices.m_numVertices; ++i)
	{
		vectorsOut[i].load<4,HK_IO_NATIVE_ALIGNED>(&va[4*i]);
	}

    vertexBuffer->unlock(lockedVertices);

    return HK_SUCCESS;
}

/* static */void hkMeshVertexBufferUtil::transform(const Buffer& srcBuffer, const hkMatrix4& transform, int transformFlags, int numVertices)
{
    const hkVertexFormat::Element& ele =  srcBuffer.m_element;

    hkUint8* cur = (hkUint8*)srcBuffer.m_start;
    const int stride = srcBuffer.m_stride;

    // Do the interpolation
    switch (ele.m_usage)
    {
        case hkVertexFormat::USAGE_POSITION:
        {
            HK_ASSERT(0x277abd2f, ele.m_dataType == hkVertexFormat::TYPE_FLOAT32 && ele.m_numValues >= 3);
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkFloat32* v = (hkFloat32*)cur;
                hkVector4 p; 
				p.load<3,HK_IO_NATIVE_ALIGNED>(v);
				hkVector4 pt;
				transform.transformPosition(p, pt);
                pt.store<3,HK_IO_NATIVE_ALIGNED>(v);
			}
            break;
        }
        case hkVertexFormat::USAGE_TANGENT:
        {
            HK_ASSERT(0x277abd2e, ele.m_dataType == hkVertexFormat::TYPE_FLOAT32 && ele.m_numValues >= 3);
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkFloat32* v = (hkFloat32*)cur;
                hkVector4 p; 
				p.load<3,HK_IO_NATIVE_ALIGNED>(v);
				hkVector4 pt;
				transform.transformDirection(p, pt);
                if (transformFlags & TRANSFORM_NORMALIZE)
                {
                    pt.normalizeIfNotZero<3>();
                }
                pt.store<3,HK_IO_NATIVE_ALIGNED>(v);
			}
            break;
        }
        case hkVertexFormat::USAGE_NORMAL:
        case hkVertexFormat::USAGE_BINORMAL:
        {
            HK_ASSERT(0x277abd2d, ele.m_dataType == hkVertexFormat::TYPE_FLOAT32 && ele.m_numValues >= 3);
            for (int i = 0; i < numVertices; i++, cur += stride)
            {
                hkFloat32* v = (hkFloat32*)cur;
                hkVector4 p; 
				p.load<3,HK_IO_NATIVE_ALIGNED>(v);
				if (transformFlags & TRANSFORM_PRE_NEGATE)
                {
                    p.setNeg<4>(p);
                }
				hkVector4 pt;
                transform.transformDirection(p, pt);
                if (transformFlags& TRANSFORM_POST_NEGATE)
                {
                    pt.setNeg<4>(pt);
                }
                if (transformFlags & TRANSFORM_NORMALIZE)
                {
                    pt.normalizeIfNotZero<3>();
                }
                pt.store<3,HK_IO_NATIVE_ALIGNED>(v);
			}
            break;
        }
		
        default: break;
    }
}

static void hkMeshVertexBufferUtil_interpolateCopy(const hkVertexFormat::Element& element, const void* a, const void* b, hkSimdRealParameter interp, void* dst)
{
	// Copy - a if < 0.5, b >= 0.5f
	int size = element.calculateAlignedSize();

	if (interp.isLess(hkSimdReal_Half))
	{
		hkString::memCpy4(dst, a, size / 4);
	}
	else
	{
		hkString::memCpy(dst, b, size / 4);
	}
}

static void hkMeshVertexBufferUtil_interpolate(const hkVertexFormat::Element& element, const void* aIn, const void* bIn, hkSimdRealParameter interp, void* dstIn)
{
	const int numValues = element.m_numValues;
	switch (element.m_dataType)
	{
		default:
		{
			HK_ASSERT(0x32432432, !"Unhandled type" );
			hkMeshVertexBufferUtil_interpolateCopy(element, aIn, bIn, interp, dstIn);
			break;
		}
		case hkVertexFormat::TYPE_ARGB32:
		{
			// Magic number - added to a float in the range 0-255.0f will leave the int result in the bottom 8 bits
			hkVector4 magic; magic.setAll(1 << 23);
#if defined(HK_REAL_IS_DOUBLE)
			HK_ASSERT2(0x233,false,"magic doesnt work for doubles");
#endif

			const hkUint32* a = (const hkUint32*)aIn;
			const hkUint32* b = (const hkUint32*)bIn;
			hkUint32* dst = (hkUint32*)dstIn;
			for (int i = 0; i < numValues; i++)
			{
				const hkUint32 ai = a[i];
				const hkUint32 bi = b[i];
				hkIntVector iva; iva.set( (ai >> 24) & 0xff, (ai >> 16) & 0xff, (ai >> 8) & 0xff, ai & 0xff);
				hkIntVector ivb; ivb.set( (bi >> 24) & 0xff, (bi >> 16) & 0xff, (bi >> 8) & 0xff, bi & 0xff);
				hkVector4 va; iva.convertS32ToF32(va);
				hkVector4 vb; ivb.convertS32ToF32(vb);

				hkVector4 d;
				d.setInterpolate(va, vb, interp);
				d.add(magic);

				hkIntVector v; v.loadAsFloat32BitRepresentation(d);

				dst[i] = ((v.getU32<0>() & 0xff) << 24) | ((v.getU32<1>() & 0xff) << 16) | ((v.getU32<2>() & 0xff) << 8) | ((v.getU32<3>() & 0xff) << 0);
			}
			break;
		}
		case hkVertexFormat::TYPE_FLOAT32:
		{
			const hkFloat32* a = (const hkFloat32*)aIn;
			const hkFloat32* b = (const hkFloat32*)bIn;
			hkFloat32* dst = (hkFloat32*)dstIn;

			for (int i = 0; i < numValues; i++)
			{
				hkSimdReal as; as.load<1>(&a[i]);
				hkSimdReal bs; bs.load<1>(&b[i]);
				hkSimdReal nval; nval.setInterpolate(as,bs,interp);
				nval.store<1>(&dst[i]);
			}
			break;
		}
		case hkVertexFormat::TYPE_VECTOR4:
		{
			const hkFloat32* a = (const hkFloat32*)aIn;
			const hkFloat32* b = (const hkFloat32*)bIn;
			hkFloat32* dst = (hkFloat32*)dstIn;

			for (int i = 0; i < numValues; i++)
			{
				hkVector4 va; va.load<4,HK_IO_NATIVE_ALIGNED>(&a[4*i]);
				hkVector4 vb; vb.load<4,HK_IO_NATIVE_ALIGNED>(&b[4*i]);
				hkVector4 d; d.setInterpolate(va, vb, interp);
				d.store<4,HK_IO_NATIVE_ALIGNED>(&dst[4*i]);
			}
			break;
		}
	}
}

static void hkMeshVertexBufferUtil_interpolateNormalize(const hkVertexFormat::Element& element, const void* aIn, const void* bIn, hkSimdRealParameter interp, void* dstIn)
{
	const int numValues = element.m_numValues;
	switch (element.m_dataType)
	{
		default:
		{
			// Don't know how to normalize, try to just interpolate
			hkMeshVertexBufferUtil_interpolate(element, aIn, bIn, interp, dstIn);
			break;
		}
		case hkVertexFormat::TYPE_ARGB32:
		{
			// Magic number - added to a float in the range 0-255.0f will leave the int result in the bottom 8 bits
			hkVector4 magic; magic.setAll(1 << 23);
#if defined(HK_REAL_IS_DOUBLE)
			HK_ASSERT2(0x233,false,"magic doesnt work for doubles");
#endif

			const hkUint32* a = (const hkUint32*)aIn;
			const hkUint32* b = (const hkUint32*)bIn;
			hkUint32* dst = (hkUint32*)dstIn;
			for (int i = 0; i < numValues; i++)
			{
				const hkUint32 ai = a[i];
				const hkUint32 bi = b[i];
				hkIntVector iva; iva.set( ai & 0xff, (ai >> 8) & 0xff, (ai >> 16) & 0xff, (ai >> 24) & 0xff);
				hkIntVector ivb; ivb.set( bi & 0xff, (bi >> 8) & 0xff, (bi >> 16) & 0xff, (bi >> 24) & 0xff);
				hkVector4 va; iva.convertS32ToF32(va);
				hkVector4 vb; ivb.convertS32ToF32(vb);

				hkVector4 d;
				d.setInterpolate(va, vb, interp);
				hkVector4 n = d;
				const hkSimdReal len = n.lengthSquared<3>();
				if (len.isGreater(hkSimdReal::fromFloat(hkReal(1e-6f)*hkReal(1e-6f))))
				{
					n.mul( hkSimdReal_255 * len.sqrtInverse() );	
				}

				d.setXYZ_W(n, d);
				d.add(magic);

				hkIntVector v; v.loadAsFloat32BitRepresentation(d);

				dst[i] = ((v.getU32<3>() & 0xff) << 24) | ((v.getU32<2>() & 0xff) << 16) | ((v.getU32<1>() & 0xff) << 8) | ((v.getU32<0>() & 0xff) << 0);
			}
			break;
		}
		case hkVertexFormat::TYPE_FLOAT32:
		{
			switch (numValues)
			{
				case 3:
				{
					hkVector4 a,b;
					a.load<3,HK_IO_NATIVE_ALIGNED>((const hkFloat32*)aIn);
					b.load<3,HK_IO_NATIVE_ALIGNED>((const hkFloat32*)bIn);
					hkVector4 d; d.setInterpolate(a, b, interp);
					d.normalize<3>();
					hkFloat32* v = (hkFloat32*)dstIn;
					d.store<3,HK_IO_NATIVE_ALIGNED>(v);
					break;
				}
				case 4:
				{
					hkVector4 a,b;
					a.load<4,HK_IO_NATIVE_ALIGNED>((const hkFloat32*)aIn);
					b.load<4,HK_IO_NATIVE_ALIGNED>((const hkFloat32*)bIn);
					hkVector4 d; d.setInterpolate(a, b, interp);
					d.normalize<4>();
					hkFloat32* v = (hkFloat32*)dstIn;
					d.store<4,HK_IO_NATIVE_ALIGNED>(v);
				}
				default:
				{
					// Can't normalize unless its a 3 or 4... so just interpolate
					hkMeshVertexBufferUtil_interpolate(element, aIn, bIn, interp, dstIn);
					break;
				}
			}
			break;
		}
		case hkVertexFormat::TYPE_VECTOR4:
		{
			const hkFloat32* a = (const hkFloat32*)aIn;
			const hkFloat32* b = (const hkFloat32*)bIn;
			hkFloat32* dst = (hkFloat32*)dstIn;

			for (int i = 0; i < numValues; i++)
			{
				hkVector4 va; va.load<4,HK_IO_NATIVE_ALIGNED>(&a[4*i]);
				hkVector4 vb; vb.load<4,HK_IO_NATIVE_ALIGNED>(&b[4*i]);
				hkVector4 d; d.setInterpolate(va, vb, interp);
				d.normalizeIfNotZero<3>();
				d.store<4,HK_IO_NATIVE_ALIGNED>(&dst[4*i]);
			}
			break;
		}
	}
}

/* static */void hkMeshVertexBufferUtil::interpolate(const Element& element, const void* a, const void* b, hkSimdRealParameter interp, void* dst)
{
	switch (element.m_usage)
	{
		case hkVertexFormat::USAGE_TANGENT:
		case hkVertexFormat::USAGE_BINORMAL:
		case hkVertexFormat::USAGE_NORMAL:
		{
			hkMeshVertexBufferUtil_interpolateNormalize(element, a, b, interp, dst);
			break;
		}
		case hkVertexFormat::USAGE_POSITION:
		case hkVertexFormat::USAGE_POINT_SIZE:
		case hkVertexFormat::USAGE_TEX_COORD:
		case hkVertexFormat::USAGE_COLOR:
		{
			// Interpolate
			hkMeshVertexBufferUtil_interpolate(element, a, b, interp, dst);
			break;
		}
		case hkVertexFormat::USAGE_BLEND_MATRIX_INDEX:
		case hkVertexFormat::USAGE_BLEND_WEIGHTS:
		case hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED:
		{
			hkMeshVertexBufferUtil_interpolateCopy(element, a, b, interp, dst);
			break;			
		}
		default:
		{
			HK_ASSERT(0x31254343, !"Unknown type, unable to interpolate");
		}
	}
}

/* static */hkResult hkMeshVertexBufferUtil::transform(hkMeshVertexBuffer* buffer, const hkMatrix4& transformIn, int transformFlags)
{
    hkMeshVertexBuffer::LockedVertices lockedVertices;
	hkMeshVertexBuffer::LockInput lockInput;


    hkMeshVertexBuffer::LockResult lockRes = buffer->lock(lockInput, lockedVertices);
	if (lockRes != hkMeshVertexBuffer::RESULT_SUCCESS)
	{
		HK_ASSERT(0x277abd2c, !"Could not lock");
        return HK_FAILURE;
	}

    const int numBuffers = lockedVertices.m_numBuffers;
    for (int i = 0; i < numBuffers; i++)
    {
        transform(lockedVertices.m_buffers[i], transformIn, transformFlags, lockedVertices.m_numVertices);
    }

    buffer->unlock(lockedVertices);

    return HK_SUCCESS;
}

/* static */hkMeshVertexBuffer* hkMeshVertexBufferUtil::concatVertexBuffers(hkMeshSystem* system, hkMeshVertexBuffer** buffers, int numBuffers)
{
	if (numBuffers == 0)
	{
		return HK_NULL;
	}

	if (numBuffers == 1)
	{
		buffers[0]->addReference();
		return buffers[0];
	}

	hkVertexFormat vertexFormat;
	buffers[0]->getVertexFormat(vertexFormat);

	int totalVertices = 0;
	for (int i = 0; i < numBuffers; i++)
	{
		hkMeshVertexBuffer* buffer = buffers[i];

		hkVertexFormat bufferVertexFormat;
		buffer->getVertexFormat(bufferVertexFormat);

		if (bufferVertexFormat != vertexFormat)
		{
			// To merge they must be the same format...
			HK_ASSERT(0x42542543, !"The buffers are not the same format");
			return HK_NULL;
		}
		totalVertices += buffer->getNumVertices();
	}

	// Create the output vertex buffer

	hkMeshVertexBuffer* dstVertexBuffer = system->createVertexBuffer(vertexFormat, totalVertices);
	if (!dstVertexBuffer)
	{
		return HK_NULL;
	}

	hkMeshVertexBuffer::LockedVertices dstAllLocked;
	{
		hkMeshVertexBuffer::LockInput input;
		input.m_lockFlags = hkMeshVertexBuffer::ACCESS_WRITE | hkMeshVertexBuffer::ACCESS_WRITE_DISCARD;
		hkMeshVertexBuffer::LockResult result = dstVertexBuffer->lock(input, dstAllLocked);
		if (result != hkMeshVertexBuffer::RESULT_SUCCESS)
		{
			dstVertexBuffer->removeReference();
			return HK_NULL;
		}
	}

	{
		hkMeshVertexBuffer::LockedVertices dstLocked = dstAllLocked;

		for (int i = 0; i < numBuffers; i++)
		{
			hkMeshVertexBuffer* srcBuffer = buffers[i];

			hkMeshVertexBuffer::LockInput input;
			input.m_lockFlags = hkMeshVertexBuffer::ACCESS_READ;
			hkMeshVertexBuffer::LockedVertices srcLocked;

			// Lock the buffer to copy from
			hkMeshVertexBuffer::LockResult res = srcBuffer->lock(input, srcLocked);
			if (res != hkMeshVertexBuffer::RESULT_SUCCESS)
			{
				dstVertexBuffer->removeReference();
				return HK_NULL;
			}

			// Make the same
			dstLocked.m_numVertices = srcLocked.m_numVertices;

			// Okay we need to merge all together
			hkMeshVertexBufferUtil::copy(srcLocked, dstLocked);

			// We don't need this lock anymore -> as the data has been copied
			srcBuffer->unlock(srcLocked);

			const int numVertices = srcBuffer->getNumVertices();
			// Move along by the amount of vertices in the destination
			for (int j = 0; j < dstLocked.m_numBuffers; j++)
			{
				hkMeshVertexBuffer::LockedVertices::Buffer& buffer = dstLocked.m_buffers[j];
				buffer.m_start = ((char*)buffer.m_start) + (buffer.m_stride * numVertices);
			}
		}
	}

	// Remove the lock
	dstVertexBuffer->unlock(dstAllLocked);
	// Done
	return dstVertexBuffer;
}

/* static */hkBool32 HK_CALL hkMeshVertexBufferUtil::isBufferNormalDataEqual(const Buffer& bufferA, const Buffer& bufferB, hkSimdRealParameter threshold)
{
	const hkVertexFormat::Element& eleA = bufferA.m_element;
	const hkVertexFormat::Element& eleB = bufferB.m_element;

	if (eleA.m_dataType != eleB.m_dataType ||
		eleA.m_numValues != eleB.m_numValues)
	{
		HK_WARN(0xd8279a0b, "Cannot compare as types are different");
		return hkFalse32;
	}

	if (eleA.m_dataType != hkVertexFormat::TYPE_FLOAT32 && eleA.m_numValues != 3)
	{
		return isBufferDataEqual(bufferA, bufferB, threshold);
	}

	hkVector4 normalA; normalA.load<3,HK_IO_NATIVE_ALIGNED>((const hkFloat32*)bufferA.m_start);
	hkVector4 normalB; normalB.load<3,HK_IO_NATIVE_ALIGNED>((const hkFloat32*)bufferB.m_start);

	hkSimdReal err; err.setAbs(hkSimdReal_1 - normalA.dot<3>(normalB));

	return err.isLess(threshold);
}


/* static */hkBool32 HK_CALL hkMeshVertexBufferUtil::isBufferDataEqual(const Buffer& bufferA, const Buffer& bufferB, hkSimdRealParameter threshold)
{
    const hkVertexFormat::Element& eleA = bufferA.m_element;
    const hkVertexFormat::Element& eleB = bufferB.m_element;

    if (eleA.m_dataType != eleB.m_dataType ||
        eleA.m_numValues != eleB.m_numValues)
    {
        HK_WARN(0xd8279a0a, "Cannot compare as types are different");
        return hkFalse32;
    }

    const int numValues = eleA.m_numValues;
    // See if they are equal
    switch (eleA.m_dataType)
    {
		case hkVertexFormat::TYPE_FLOAT32:
        {
            const hkFloat32* va = (const hkFloat32*)(bufferA.m_start);
            const hkFloat32* vb = (const hkFloat32*)(bufferB.m_start);
            for (int j = 0; j < numValues; j++)
            {
				hkSimdReal vas; vas.load<1>(&va[j]);
				hkSimdReal vbs; vbs.load<1>(&vb[j]);
				hkSimdReal err; err.setAbs(vas - vbs);
                if ( err.isGreaterEqual(threshold) )
                {
                    return hkFalse32;
                }
            }
            break;
        }
		case hkVertexFormat::TYPE_ARGB32:
        {
            const hkUint32* va = (const hkUint32*)(bufferA.m_start);
            const hkUint32* vb = (const hkUint32*)(bufferB.m_start);
			// Work out the threshold
			hkInt32 intThreshold; (threshold * hkSimdReal_255).storeSaturateInt32(&intThreshold);

            for (int j = 0; j < numValues; j++)
            {
                hkUint32 a = va[j];
                hkUint32 b = vb[j];

                if (a != b)
                {
                    if (threshold.isEqualZero())
                    {
                        return hkFalse32;
                    }

                    for (int i = 0; i < 4; i++)
                    {
                        hkInt32 diff = hkInt32(a & 0xff) - hkInt32(b & 0xff);

                        // Trick to work out he absolute without a branch
                        const hkInt32 mask = diff >> 31;
                        diff = (diff + mask) ^ mask;
                        // Is it bigger than the threshold
                        if (diff > intThreshold)
                        {
                            return hkFalse32;
                        }
                        // Next byte
                        a = a >> 8;
                        b = b >> 8;
                    }
                }
            }
            break;
        }
		case hkVertexFormat::TYPE_INT32:
		case hkVertexFormat::TYPE_UINT32:
		case hkVertexFormat::TYPE_UINT8_DWORD:
        {
            const hkUint32* va = (const hkUint32*)(bufferA.m_start);
            const hkUint32* vb = (const hkUint32*)(bufferB.m_start);
            for (int j = 0; j < numValues; j++)
            {
                if (va[j] != vb[j])
                {
                    return hkFalse32;
                }
            }
            break;
        }
		case hkVertexFormat::TYPE_INT8:
		case hkVertexFormat::TYPE_UINT8:
        {
            const hkUint8* va = (const hkUint8*)(bufferA.m_start);
            const hkUint8* vb = (const hkUint8*)(bufferB.m_start);
            for (int j = 0; j < numValues; j++)
            {
                if (va[j] != vb[j])
                {
                    return hkFalse32;
                }
            }
            break;
        }
		case hkVertexFormat::TYPE_INT16:
		case hkVertexFormat::TYPE_UINT16:
        {
            const hkUint16* va = (const hkUint16*)(bufferA.m_start);
            const hkUint16* vb = (const hkUint16*)(bufferB.m_start);
            for (int j = 0; j < numValues; j++)
            {
                if (va[j] != vb[j])
                {
                    return hkFalse32;
                }
            }
            break;
        }
		case hkVertexFormat::TYPE_FLOAT16:
        {
            HK_ASSERT(0x32aa233, !"Not handled yet");
            return hkFalse32;
        }
		default:
        {
            HK_ASSERT(0xa02ef12, !"Unknown type");
            return hkFalse32;
        }
	}

    return 1;
}

/* static */hkBool32 HK_CALL hkMeshVertexBufferUtil::isBufferDataEqual(const Buffer* buffersA, const Buffer* buffersB, int numBuffers, const Thresholds& thresholds)
{
    for (int i = 0; i < numBuffers; i++)
	{
        const hkMeshVertexBuffer::LockedVertices::Buffer& bufferA = buffersA[i];
        const hkMeshVertexBuffer::LockedVertices::Buffer& bufferB = buffersB[i];

        const hkVertexFormat::Element& eleA = bufferA.m_element;
		const hkVertexFormat::Element& eleB = bufferB.m_element;

        if (eleA.m_usage != eleB.m_usage)
        {
            HK_ASSERT(0xdd827297, !"Buffers have different usage");
            return hkFalse32;
        }

		hkBool32 result;
        switch (eleA.m_usage)
        {
            case hkVertexFormat::USAGE_POINT_SIZE:
            case hkVertexFormat::USAGE_POSITION:
            {
				result = isBufferDataEqual(bufferA, bufferB, hkSimdReal::fromFloat(thresholds.m_positionThreshold));
                break;
            }
            case hkVertexFormat::USAGE_NORMAL:
            case hkVertexFormat::USAGE_BINORMAL:
            case hkVertexFormat::USAGE_TANGENT:
            {
                result = isBufferNormalDataEqual(bufferA, bufferB, hkSimdReal::fromFloat(thresholds.m_normalThreshold));
                break;
            }
			case hkVertexFormat::USAGE_COLOR:
            {
				result = isBufferDataEqual(bufferA, bufferB, hkSimdReal::fromFloat(thresholds.m_colorThreshold));
                break;
            }
            case hkVertexFormat::USAGE_TEX_COORD:
            {
				result = isBufferDataEqual(bufferA, bufferB, hkSimdReal::fromFloat(thresholds.m_texCoordThreshold));
                break;
            }
			default:
			{
				result = isBufferDataEqual(bufferA, bufferB, hkSimdReal::fromFloat(thresholds.m_otherThreshold));
				break;
			}
        }
		if (result == hkFalse32)
		{
			return hkFalse32; // early out
		}
    }

    return 1;
}

//	Checks if a buffer has skinnable data (i.e. positions, normals, tangents and / or bitangents)
/* static */bool HK_CALL hkMeshVertexBufferUtil::bufferIsSkinnable(hkMeshVertexBuffer* vertexBuffer)
{
	// Get the vertex buffer format
	hkVertexFormat vertexFmt;
	vertexBuffer->getVertexFormat(vertexFmt);

	// See what to skin: positions, normals, binormals and tangents
	return	(vertexFmt.findElementIndex(hkVertexFormat::USAGE_POSITION, 0)	>= 0)	||
		(vertexFmt.findElementIndex(hkVertexFormat::USAGE_NORMAL, 0)	>= 0)	||
		(vertexFmt.findElementIndex(hkVertexFormat::USAGE_TANGENT, 0)	>= 0)	||
		(vertexFmt.findElementIndex(hkVertexFormat::USAGE_BINORMAL, 0)	>= 0);
}

//	Checks if a buffer has skin weights info
/* static */bool HK_CALL hkMeshVertexBufferUtil::bufferHasWeights(hkMeshVertexBuffer* vertexBuffer)
{
	// Get vertex format
	hkVertexFormat vertexFmt;
	vertexBuffer->getVertexFormat(vertexFmt);

	// Get the bone indices & weights if any
	int idxBoneWeights	= vertexFmt.findElementIndex(hkVertexFormat::USAGE_BLEND_WEIGHTS, 0);
	int idxBoneIndices	= vertexFmt.findElementIndex(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, 0);
	if ( idxBoneIndices < 0 )
	{
		idxBoneIndices = vertexFmt.findElementIndex(hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED, 0);
	}

	// Nothing to skin!
	if ( (idxBoneWeights < 0) || (idxBoneIndices < 0) )
	{
		return false;
	}

	// Skinning data available!
	return true;
}

//
//	Merges the source vertex format into the destination vertex format

void HK_CALL hkMeshVertexBufferUtil::mergeVertexFormat(hkVertexFormat& dstFmt, const hkVertexFormat& srcFmt)
{
	// Merge all elements in the source
	const int numSourceElements = srcFmt.m_numElements;
	for (int sei = 0; sei < numSourceElements; sei++)
	{
		const hkVertexFormat::Element& srcElt = srcFmt.m_elements[sei];

		// See if we have the same element in the destination already
		int dei = dstFmt.findElementIndex(srcElt.m_usage, srcElt.m_subUsage);
		if ( dei < 0 )
		{
			// Element not found, add now!
			hkVertexFormat::Element& dstElt = dstFmt.m_elements[dstFmt.m_numElements++];
			dstElt = srcElt;
		}
		else
		{
			// Element already present. Pick the highest number of components and the biggest storage
			hkVertexFormat::Element& dstElt = dstFmt.m_elements[dei];

			dstElt.m_numValues	= hkMath::max2(dstElt.m_numValues, srcElt.m_numValues);
			dstElt.m_dataType	= (hkVertexFormat::ComponentType)hkMath::max2(dstElt.m_dataType, srcElt.m_dataType);
		}
	}
}

//
//	Computes the most fitting vertex format, that will be able to store all data in the given source vertex formats

void HK_CALL hkMeshVertexBufferUtil::computeMostFittingVertexFormat(hkVertexFormat& dstFormat, const hkVertexFormat* srcFormats, int numSourceFormats)
{
	dstFormat.m_numElements = 0;
	for (int k = 0; k < numSourceFormats; k++)
	{
		mergeVertexFormat(dstFormat, srcFormats[k]);
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
