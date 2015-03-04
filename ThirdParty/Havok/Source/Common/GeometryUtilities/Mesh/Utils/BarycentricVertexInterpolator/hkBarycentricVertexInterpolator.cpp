/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Math/Matrix/hkMatrix4Util.h>
#include <Common/GeometryUtilities/Mesh/Utils/BarycentricVertexInterpolator/hkBarycentricVertexInterpolator.h>

#include <Geometry/Collide/Algorithms/Triangle/hkcdTriangleUtil.h>

hkBarycentricVertexInterpolator::hkBarycentricVertexInterpolator()
:   m_isStarted(false)
{
}

/* static */void HK_CALL hkBarycentricVertexInterpolator::calcBarycentricCoordinates(hkVector4Parameter pos, hkVector4Parameter t0, hkVector4Parameter t1, hkVector4Parameter t2, hkVector4& lambdas)
{
    hkVector4 R, Q;
    Q.setSub(t0, t1);
    R.setSub(t2, t1);

    const hkSimdReal QQ = Q.lengthSquared<3>();
    const hkSimdReal RR = R.lengthSquared<3>();
    const hkSimdReal QR = R.dot<3>(Q);

    const hkSimdReal QQRR = QQ * RR;
    const hkSimdReal QRQR = QR * QR;
    const hkSimdReal Det = (QQRR - QRQR);

	if ( Det.isGreaterZero() )
	{
		hkSimdReal invDet; invDet.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(Det);

		hkVector4 relPos; relPos.setSub( t1, pos );
		const hkSimdReal sq = relPos.dot<3>(Q);
		const hkSimdReal sr = relPos.dot<3>(R);

		const hkSimdReal q = (sr * QR - RR * sq);
		const hkSimdReal r = (sq * QR - QQ * sr);

		lambdas.set(q,
					Det - q - r,
					r,
					hkSimdReal_0);
		lambdas.mul(invDet);
		return;
	}

	hkVector4 S;
	S.setSub( t0, t2 );
	const hkSimdReal SS = S.lengthSquared<3>();

	if ( QQ.isGreaterEqual(RR) )
	{
		if ( SS.isGreaterEqual(QQ) )
		{
			if ( SS.isGreaterZero() )
			{
				hkVector4 relPos; relPos.setSub( pos, t2 );
				const hkSimdReal p = relPos.dot<3>(S);
				hkSimdReal quot; quot.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(p,SS);
				lambdas.set(quot,
							hkSimdReal_0,
							hkSimdReal_1 - quot,
							hkSimdReal_0);
			}
			else
			{
				lambdas = hkVector4::getConstant<HK_QUADREAL_0010>();
			}
		}
		else
		{
			hkVector4 relPos; relPos.setSub( pos, t1 );
			const hkSimdReal p  = relPos.dot<3>(Q);
			hkSimdReal quot; quot.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(p,QQ);
			lambdas.set(quot,
						hkSimdReal_1 - quot,
						hkSimdReal_0,
						hkSimdReal_0);
		}
	}
	else
	{
		if ( SS.isGreaterEqual(RR) )
		{
			hkVector4 relPos; relPos.setSub( pos, t2 );
			const hkSimdReal p = relPos.dot<3>(S);
			hkSimdReal quot; quot.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(p,SS);
			lambdas.set(quot,
						hkSimdReal_0,
						hkSimdReal_1 - quot,
						hkSimdReal_0);
		}
		else
		{
			hkVector4 relPos; relPos.setSub( pos, t1 );
			const hkSimdReal p  = relPos.dot<3>(R);
			hkSimdReal quot; quot.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(p,RR);
			lambdas.set(hkSimdReal_0,
						hkSimdReal_1 - quot,
						quot,
						hkSimdReal_0);
		}
	}
}

//
//	Extracts the vertices of the given triangle from the given buffer

static HK_FORCE_INLINE void HK_CALL hkBarycentricVertexInterpolator_getSupportVectors(	const hkMeshVertexBuffer::LockedVertices::Buffer& buffer,
																						const int supportIndices[3],
																						hkVector4& vA, hkVector4& vB, hkVector4& vC)
{
	HK_ASSERT(0x342323, buffer.m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32);

	const hkFloat32* s0 = (const hkFloat32*)hkAddByteOffsetConst(buffer.m_start, buffer.m_stride * supportIndices[0]);
	const hkFloat32* s1 = (const hkFloat32*)hkAddByteOffsetConst(buffer.m_start, buffer.m_stride * supportIndices[1]);
	const hkFloat32* s2 = (const hkFloat32*)hkAddByteOffsetConst(buffer.m_start, buffer.m_stride * supportIndices[2]);

	switch ( buffer.m_element.m_numValues )
	{
		case 1:		
			{
				vA.setZero(); hkSimdReal a; a.load<1>(s0); vA.setComponent<0>(a);
				vB.setZero(); hkSimdReal b; b.load<1>(s1); vB.setComponent<0>(b);
				vC.setZero(); hkSimdReal c; c.load<1>(s2); vC.setComponent<0>(c);
			}
			break;
		case 2:		
			{
				vA.load<2,HK_IO_NATIVE_ALIGNED>(s0); vA.zeroComponent<2>(); vA.zeroComponent<3>();
				vB.load<2,HK_IO_NATIVE_ALIGNED>(s1); vB.zeroComponent<2>(); vB.zeroComponent<3>();
				vC.load<2,HK_IO_NATIVE_ALIGNED>(s2); vC.zeroComponent<2>(); vC.zeroComponent<3>();
			}
			break;
		case 3:		
			{
				vA.load<3,HK_IO_NATIVE_ALIGNED>(s0); vA.zeroComponent<3>();
				vB.load<3,HK_IO_NATIVE_ALIGNED>(s1); vB.zeroComponent<3>();
				vC.load<3,HK_IO_NATIVE_ALIGNED>(s2); vC.zeroComponent<3>();
			}
			break;
		default:	
			{
				vA.load<4,HK_IO_NATIVE_ALIGNED>(s0);	
				vB.load<4,HK_IO_NATIVE_ALIGNED>(s1);	
				vC.load<4,HK_IO_NATIVE_ALIGNED>(s2);	
			}
			break;
	}
}

//
//	Interpolates a vertex across the given triangle at the given barycentric coordinates

static HK_FORCE_INLINE void hkBarycentricVertexInterpolator_interpolateValue(	const hkMeshVertexBuffer::LockedVertices::Buffer& buffer,
																				hkVector4Parameter lambdas, const int supportIndices[3],
																				hkVector4& valueOut)
{
    hkVector4 vA, vB, vC;
	hkBarycentricVertexInterpolator_getSupportVectors(buffer, supportIndices, vA, vB, vC);

	vA.			mul(				lambdas.getComponent<0>());
    vA.			addMul(vB,			lambdas.getComponent<1>());
	valueOut.	setAddMul(vA, vC,	lambdas.getComponent<2>());
}

static HK_FORCE_INLINE hkUint32 blend(hkVector4Parameter lambdas, hkUint32 s0, hkUint32 s1, hkUint32 s2)
{
	hkIntVector iv; iv.set(s0,s1,s2,0);
	hkVector4 fv; iv.convertS32ToF32(fv);
	hkUint32 c = 0;
	(fv.dot<3>(lambdas) + hkSimdReal_Half).storeSaturateInt32((hkInt32*)&c);
	return hkMath::min2(c, hkUint32(0xff));
}

// interpolate 8bit RGBA color
static HK_FORCE_INLINE void hkBarycentricVertexInterpolator_interpolateColor(const hkMeshVertexBuffer::LockedVertices::Buffer& buffer, hkVector4Parameter lambdas, const int supportIndices[3], hkUint32& colorOut)
{
	HK_ON_DEBUG(const hkVertexFormat::Element& ele = buffer.m_element;)
	HK_ASSERT(0x342323, ele.m_dataType == hkVertexFormat::TYPE_ARGB32);
	HK_ASSERT(0x342333, ele.m_numValues == 1);

	const hkUint32 s0 = *((const hkUint32*)(((const char*)buffer.m_start) + buffer.m_stride * supportIndices[0]));
	const hkUint32 s1 = *((const hkUint32*)(((const char*)buffer.m_start) + buffer.m_stride * supportIndices[1]));
	const hkUint32 s2 = *((const hkUint32*)(((const char*)buffer.m_start) + buffer.m_stride * supportIndices[2]));

	// per color channel interpolation
	hkUint32 A = blend(lambdas, ((s0 >> 24) & 0xff), ((s1 >> 24) & 0xff), ((s2 >> 24) & 0xff));
	hkUint32 R = blend(lambdas, ((s0 >> 16) & 0xff), ((s1 >> 16) & 0xff), ((s2 >> 16) & 0xff));
	hkUint32 G = blend(lambdas, ((s0 >>  8) & 0xff), ((s1 >>  8) & 0xff), ((s2 >>  8) & 0xff));
	hkUint32 B = blend(lambdas, ( s0        & 0xff), ( s1        & 0xff), ( s2        & 0xff));

	colorOut = (A << 24) | (R << 16) | (G << 8) | B;
}

static HK_FORCE_INLINE void hkBarycentricVertexInterpolator_storeValue(const hkVertexFormat::Element& ele, const hkVector4& value, void* dstIn)
{
    HK_ASSERT(0x342323, ele.m_dataType == hkVertexFormat::TYPE_FLOAT32);

    hkFloat32* dst = (hkFloat32*)dstIn;
    switch (ele.m_numValues)
    {
        case 1:
        {
			value.store<1,HK_IO_NATIVE_ALIGNED>(dst);
            break;
        }
        case 2:
        {
			value.store<2,HK_IO_NATIVE_ALIGNED>(dst);
            break;
        }
        case 3:
        {
			value.store<3,HK_IO_NATIVE_ALIGNED>(dst);
            break;
        }
        default:
        {
			value.store<4,HK_IO_NATIVE_ALIGNED>(dst);
            break;
        }
    }
}

static HK_FORCE_INLINE void hkBarycentricVertexInterpolator_loadValue(const hkVertexFormat::Element& ele, const void* srcIn, hkVector4& valueOut)
{
    HK_ASSERT(0x342323, ele.m_dataType == hkVertexFormat::TYPE_FLOAT32);

    hkFloat32* src = (hkFloat32*)srcIn;
    switch (ele.m_numValues)
    {
        case 1:
        {
			hkSimdReal a; a.load<1>(src);
			valueOut = hkVector4::getConstant<HK_QUADREAL_0001>();
			valueOut.setComponent<0>(a);
            break;
        }
        case 2:
        {
			hkVector4 a; a.load<2,HK_IO_NATIVE_ALIGNED>(src);
			valueOut.setSelect<hkVector4ComparisonMask::MASK_XY>(a,hkVector4::getConstant<HK_QUADREAL_0001>());
            break;
        }
        case 3:
        {
            valueOut.load<3,HK_IO_NATIVE_ALIGNED>(src);
			valueOut.setComponent<3>(hkSimdReal_1);
            break;
        }
        default:
        {
			valueOut.load<4,HK_IO_NATIVE_ALIGNED>(src);
            break;
        }
    }
}


static HK_FORCE_INLINE void hkBarycentricVertexInterpolator_copyValue(const hkVertexFormat::Element& ele, const void* srcIn, void* dstIn)
{
    int size = ele.calculateAlignedSize();

    if ((size & 3) == 0)
    {
        const hkUint32* src = (const hkUint32*)srcIn;
        hkUint32* dst = (hkUint32*)dstIn;

        switch (size >> 2)
        {
            case 0:
            {
                dst[0] = src[0];
                return;
            }
            case 1:
            {
                dst[0] = src[0];
                dst[1] = src[1];
                return;
            }
            case 2:
            {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                return;
            }
            default:
            {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
                return;
            }
        }
    }
    // All else fails, just string copy...
    hkString::memCpy(dstIn, srcIn, size);
}

void hkBarycentricVertexInterpolator::start(const hkMeshVertexBuffer::LockedVertices& srcLockedVertices, const hkMatrix4& objectToWorld, hkBool isInverted)
{
    if (m_isStarted)
    {
        HK_ASSERT(0x242342, !"Start has already been called on the calculator");
        // Can't continue...
        HK_BREAKPOINT(0);
        return;
    }

    // Save off
    m_srcLockedVertices = srcLockedVertices;

    // Find the position layout info for src positions
    {
        const int positionIndex = srcLockedVertices.findBufferIndex(hkVertexFormat::USAGE_POSITION, 0);
        if (positionIndex < 0)
        {
            HK_ASSERT(0x302ef142, !"Cannot find position information in source objects vertex buffer");
            // Can't continue...
            HK_BREAKPOINT(0);
            return;
        }
        m_srcPositionBuffer = srcLockedVertices.m_buffers[positionIndex];
    }

    // We need a matrix to get from world space to the local space of the object we are interpolating from
    m_objectToWorld = objectToWorld;
    hkMatrix4Util::setInverse(m_objectToWorld, m_worldToObject, hkSimdReal_Eps);

    // If this is set then we need to flip the direction of normals
    m_isInverted = isInverted;

    // Store that its in use
    m_isStarted = true;
}

void hkBarycentricVertexInterpolator::end()
{
    HK_ASSERT(0xd8279a0c, m_isStarted);
    m_isStarted = false;
}

void hkBarycentricVertexInterpolator::setSupport(int v0, int v1, int v2)
{
    HK_ASSERT(0x8d7292c3, m_isStarted);

	const int numVertices = m_srcLockedVertices.m_numVertices;

	if ( v0 >= numVertices ){ v0 = v0 % numVertices; }	// probably hitting the extruded triangles
	if ( v1 >= numVertices ){ v1 = v1 % numVertices; }	// probably hitting the extruded triangles
	if ( v2 >= numVertices ){ v2 = v2 % numVertices; }	// probably hitting the extruded triangles


    m_supportIndices[0] = v0;
    m_supportIndices[1] = v1;
    m_supportIndices[2] = v2;

    // Load the support positions
	HK_ASSERT(0x2342525f, m_srcPositionBuffer.m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32);
	const hkFloat32* s = (const hkFloat32*)m_srcPositionBuffer.m_start;
    m_supportPositions[0].load<3,HK_IO_NATIVE_ALIGNED>( hkAddByteOffsetConst( s, m_srcPositionBuffer.m_stride * v0));
    m_supportPositions[1].load<3,HK_IO_NATIVE_ALIGNED>( hkAddByteOffsetConst( s, m_srcPositionBuffer.m_stride * v1));
    m_supportPositions[2].load<3,HK_IO_NATIVE_ALIGNED>( hkAddByteOffsetConst( s, m_srcPositionBuffer.m_stride * v2));
}

//
//	Computes the support normal

void hkBarycentricVertexInterpolator::computeSupportNormal(hkVector4& normalOut) const
{
	hkVector4 vN;
	hkcdTriangleUtil::calcNonUnitNormal(m_supportPositions[0], m_supportPositions[1], m_supportPositions[2], vN);
	m_objectToWorld.transformDirection(vN, vN);

	// Safe normalize.
	const hkSimdReal len	= vN.length<3>();
	hkSimdReal invLen;		invLen.setReciprocal<HK_ACC_FULL, HK_DIV_SET_ZERO>(len);
	vN.mul(invLen);
	vN.setSelect(len.less(hkSimdReal_Eps), hkVector4::getConstant<HK_QUADREAL_1000>(), vN);	
}

//
//	Checks whether the given matrix is orthonormal

static HK_FORCE_INLINE bool HK_CALL hkBarycentricVertexInterpolator_checkOrtho(const hkRotation& r)
{
	const hkVector4& vX = r.getColumn<0>();
	const hkVector4& vY = r.getColumn<1>();
	const hkVector4& vZ = r.getColumn<2>();

	// Check orthogonality
	hkVector4 vDots;
	hkVector4Util::dot3_3vs3(vX, vY, vY, vZ, vZ, vX, vDots);
	vDots.setAbs(vDots);
	const hkSimdReal maxDot = vDots.horizontalMax<3>();
	if ( maxDot.isGreater(hkSimdReal::fromFloat(1.0e-4f)) )
	{
		HK_WARN_ONCE(0xabba0248, "Basis is not orthogonal!");
		return false;
	}

	// Check winding
	hkVector4 vCross;
	vCross.setCross(vX, vY);
	const hkSimdReal proj = vZ.dot<3>(vCross);
	if ( proj.isLessZero() )
	{
		HK_WARN_ONCE(0xabba0248, "Basis is flipped!");
		return false;
	}

	return true;
}

static HK_FORCE_INLINE const hkVector4Comparison HK_CALL hkBarycentricVertexInterpolator_makeTBN(	const hkMatrix4& worldFromObject,
																									hkVector4Parameter vSrcT, hkVector4Parameter vSrcB, hkVector4Parameter vSrcN,
																									hkQuaternion& qOut)
{
	// Transform
	hkVector4 vT;	worldFromObject.transformDirection(vSrcT, vT);	vT.normalize<3>();
	hkVector4 vB;	worldFromObject.transformDirection(vSrcB, vB);	vB.normalize<3>();
	hkVector4 vN;	worldFromObject.transformDirection(vSrcN, vN);	vN.normalize<3>();

	// Build basis
	hkVector4 vExpectedB;	vExpectedB.setCross(vN, vT);
	hkVector4 vNegB;		vNegB.setNeg<3>(vB);
	const hkSimdReal proj = vExpectedB.dot<3>(vB);
	const hkVector4Comparison cmp = proj.lessZero();
	vExpectedB.setSelect(cmp, vNegB, vB);
	
	// Convert to quaternion
	hkRotation tbn;
	tbn.setCols(vT, vExpectedB, vN);
#ifdef HK_DEBUG
	hkBarycentricVertexInterpolator_checkOrtho(tbn);
#endif
	qOut.setAndNormalize(tbn);

	return cmp;
}
void hkBarycentricVertexInterpolator::calculateVertex(const hkVector4* vertexPosition, hkVector4Parameter lambdas, const hkMeshVertexBuffer::LockedVertices& dstVertex)
{
    HK_ASSERT(0x23403423, m_isStarted);

	// Check if we have TNB
	const int idxT = m_srcLockedVertices.findBufferIndex(hkVertexFormat::USAGE_TANGENT, 0);
	const int idxN = m_srcLockedVertices.findBufferIndex(hkVertexFormat::USAGE_NORMAL, 0);
	const int idxB = m_srcLockedVertices.findBufferIndex(hkVertexFormat::USAGE_BINORMAL, 0);

	// Interpolate TNB as a whole, to maintain orthogonality
	bool hasTNB = (idxT >= 0) && (idxN >= 0) && (idxB >= 0);
	if ( hasTNB )
	{
		// Get source vectors
		hkVector4 vT0, vT1, vT2;	hkBarycentricVertexInterpolator_getSupportVectors(m_srcLockedVertices.m_buffers[idxT], m_supportIndices, vT0, vT1, vT2);
		hkVector4 vB0, vB1, vB2;	hkBarycentricVertexInterpolator_getSupportVectors(m_srcLockedVertices.m_buffers[idxB], m_supportIndices, vB0, vB1, vB2);
		hkVector4 vN0, vN1, vN2;	hkBarycentricVertexInterpolator_getSupportVectors(m_srcLockedVertices.m_buffers[idxN], m_supportIndices, vN0, vN1, vN2);
		
		// Invert if necessary
		if ( m_isInverted )
		{
			vN0.setNeg<3>(vN0);		vN1.setNeg<3>(vN1);		vN2.setNeg<3>(vN2);
			vB0.setNeg<3>(vB0);		vB1.setNeg<3>(vB1);		vB2.setNeg<3>(vB2);
		}

		// Create rotations from them
		hkQuaternion qVerts[3];
		hkVector4Comparison flip0 = hkBarycentricVertexInterpolator_makeTBN(m_objectToWorld, vT0, vB0, vN0, qVerts[0]);
		hkVector4Comparison flip1 = hkBarycentricVertexInterpolator_makeTBN(m_objectToWorld, vT1, vB1, vN1, qVerts[1]);
		hkVector4Comparison flip2 = hkBarycentricVertexInterpolator_makeTBN(m_objectToWorld, vT2, vB2, vN2, qVerts[2]);

		// See if all have the same winding
		hkVector4Comparison flip012;	flip012.setAnd(flip0, flip1);	flip012.setAnd(flip012, flip2);
		flip0.setXor(flip012, flip0);	flip1.setXor(flip012, flip1);	flip2.setXor(flip012, flip2);
		hkVector4Comparison cmp012;		cmp012.setOr(flip0, flip1);		cmp012.setOr(cmp012, flip2);
		if ( cmp012.allAreSet() )
		{
			hasTNB = false;
		}

		// Store interpolated value
		if ( hasTNB )
		{
			// Interpolate
			hkQuaternion qI;	qI.setBarycentric(qVerts, lambdas);
			hkRotation rotI;	rotI.set(qI);

			hkVector4 vInterpB	= rotI.getColumn<1>();
			hkVector4 vNegB;	vNegB.setNeg<3>(vInterpB);
			vInterpB.setSelect(flip012, vNegB, vInterpB);

			hkBarycentricVertexInterpolator_storeValue(dstVertex.m_buffers[idxT].m_element, rotI.getColumn<0>(),	dstVertex.m_buffers[idxT].m_start);
			hkBarycentricVertexInterpolator_storeValue(dstVertex.m_buffers[idxB].m_element, vInterpB,				dstVertex.m_buffers[idxB].m_start);
			hkBarycentricVertexInterpolator_storeValue(dstVertex.m_buffers[idxN].m_element, rotI.getColumn<2>(),	dstVertex.m_buffers[idxN].m_start);
		}
	}

    //
    const int numElements = m_srcLockedVertices.m_numBuffers;
    // Interpolate all of the components....
    for (int i = 0; i < numElements; i++)
    {
        const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer = dstVertex.m_buffers[i];
        const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer = m_srcLockedVertices.m_buffers[i];

        // Check they are the same format
        HK_ASSERT(0x4234324, dstBuffer.m_element == srcBuffer.m_element);

        const hkVertexFormat::Element& ele =  srcBuffer.m_element;

        // Do the interpolation
        switch (ele.m_usage)
        {
            default:
            case hkVertexFormat::USAGE_BLEND_MATRIX_INDEX:
            case hkVertexFormat::USAGE_BLEND_WEIGHTS:
            case hkVertexFormat::USAGE_POINT_SIZE:
            {
                // Just copy.. from the first support
                const char* srcValue = ((const char*)srcBuffer.m_start) + srcBuffer.m_stride * m_supportIndices[0];
                hkBarycentricVertexInterpolator_copyValue(ele, srcValue, dstBuffer.m_start);
                break;
            }
            case hkVertexFormat::USAGE_POSITION:
            {
                if (vertexPosition)
                {
                    // Just store the position
					HK_ASSERT(0x2342505f, dstBuffer.m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32);
                    vertexPosition->store<3,HK_IO_NATIVE_ALIGNED>((hkFloat32*)dstBuffer.m_start);
                }
                else
                {
                    // Need to interpolate the position
                    hkVector4 value;
                    // Interpolate the value
                    hkBarycentricVertexInterpolator_interpolateValue(srcBuffer, lambdas, m_supportIndices, value);
                    m_objectToWorld.multiplyVector(value, value);
                    // Store the value
                    hkBarycentricVertexInterpolator_storeValue(ele, value, dstBuffer.m_start);
                }
                break;
            }
			case hkVertexFormat::USAGE_COLOR:
			{
				if ( (ele.m_dataType == hkVertexFormat::TYPE_ARGB32) ||
					 ((ele.m_dataType == hkVertexFormat::TYPE_UINT8) && (ele.m_numValues == 4)) )
				{
					hkMeshVertexBuffer::LockedVertices::Buffer sb = srcBuffer;
					sb.m_element.m_dataType		= hkVertexFormat::TYPE_ARGB32;	// Assume arbg32 even if we're given 4 Uint8 values!
					sb.m_element.m_numValues	= 1;

					hkUint32 value;
					// Interpolate the value
					hkBarycentricVertexInterpolator_interpolateColor(sb, lambdas, m_supportIndices, value);
					// Store the value
					hkUint32* dst = (hkUint32*)(dstBuffer.m_start);
					*dst = value;
					break;
				}
				// else fallthrough for float data interpolation
			}
            case hkVertexFormat::USAGE_TEX_COORD:
            {
                hkVector4 value;
                // Interpolate the value
                hkBarycentricVertexInterpolator_interpolateValue(srcBuffer, lambdas, m_supportIndices, value);
                // Store the value
                hkBarycentricVertexInterpolator_storeValue(ele, value, dstBuffer.m_start);
                break;
            }
            case hkVertexFormat::USAGE_TANGENT:
                if ( !hasTNB )
				{
					hkVector4 value;
	                // Interpolate the value
	                hkBarycentricVertexInterpolator_interpolateValue(srcBuffer, lambdas, m_supportIndices, value);
	                m_objectToWorld.transformDirection(value, value);
	                value.normalizeIfNotZero<3>();

	                // Store the value
	                hkBarycentricVertexInterpolator_storeValue(ele, value, dstBuffer.m_start);
				}
				break;
				
            case hkVertexFormat::USAGE_NORMAL:
            case hkVertexFormat::USAGE_BINORMAL:
				if ( !hasTNB )
	            {
	                hkVector4 value;
	                // Interpolate the value
	                hkBarycentricVertexInterpolator_interpolateValue(srcBuffer, lambdas, m_supportIndices, value);

	                m_objectToWorld.transformDirection(value, value);
	                // We need to normalize because the transform may scale
	                value.normalizeIfNotZero<3>();

	                if (m_isInverted)
	                {
	                    // Flip the direction
	                    value.setNeg<4>(value);
	                }

	                // Store the value
	                hkBarycentricVertexInterpolator_storeValue(ele, value, dstBuffer.m_start);
				}
                break;
        }
    }
}


void hkBarycentricVertexInterpolator::calculateVertex(const hkVector4& vertexPosition, const hkMeshVertexBuffer::LockedVertices& dstVertex)
{
    HK_ASSERT(0x23c23423, m_isStarted);

    // Calculate barycentric coordinates
    hkVector4 objectVertexPosition;
    m_worldToObject.transformPosition(vertexPosition, objectVertexPosition);

    hkVector4 lambdas;
    // Calculate barycentric coords for the vertex position
    calcBarycentricCoordinates(objectVertexPosition, m_supportPositions[0], m_supportPositions[1], m_supportPositions[2], lambdas);

    calculateVertex(&vertexPosition, lambdas, dstVertex);
}

void hkBarycentricVertexInterpolator::copyVertex(int vertexIndex, const hkMeshVertexBuffer::LockedVertices& dstVertex)
{
    HK_ASSERT(0x8d7292c2, m_isStarted);
    //
    const int numElements = m_srcLockedVertices.m_numBuffers;
    // Interpolate all of the components....
    for (int i = 0; i < numElements; i++)
    {
        const hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer = dstVertex.m_buffers[i];
        const hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer = m_srcLockedVertices.m_buffers[i];

        // Check they are the same format
        HK_ASSERT(0x4234324, dstBuffer.m_element == srcBuffer.m_element);

        const char* srcValue = ((const char*)srcBuffer.m_start) + srcBuffer.m_stride * vertexIndex;

        const hkVertexFormat::Element& ele =  srcBuffer.m_element;

        // Do the interpolation
        switch (ele.m_usage)
        {
            default:
            case hkVertexFormat::USAGE_BLEND_MATRIX_INDEX:
            case hkVertexFormat::USAGE_BLEND_WEIGHTS:
            case hkVertexFormat::USAGE_POINT_SIZE:
            {
                // Just copy
                hkBarycentricVertexInterpolator_copyValue(ele, srcValue, dstBuffer.m_start);
                break;
            }
            case hkVertexFormat::USAGE_POSITION:
            {
                hkVector4 value;
                hkBarycentricVertexInterpolator_loadValue(ele, srcValue, value);
                m_objectToWorld.multiplyVector(value, value);
                hkBarycentricVertexInterpolator_storeValue(ele, value, dstBuffer.m_start);
                break;
            }
			case hkVertexFormat::USAGE_COLOR:
			{
				if (ele.m_dataType == hkVertexFormat::TYPE_ARGB32)
				{
					HK_ASSERT(0x425ba0, ele.m_numValues == 1);
					hkUint32* src = (hkUint32*)srcValue;
					hkUint32* dst = (hkUint32*)(dstBuffer.m_start);
					*dst = *src;
				}
				// else fallthrough float case
			}
            case hkVertexFormat::USAGE_TEX_COORD:
            {
                // Just copy
                hkBarycentricVertexInterpolator_copyValue(ele, srcValue, dstBuffer.m_start);
                break;
            }
            case hkVertexFormat::USAGE_TANGENT:
            {
                hkVector4 value;
                // Load the value
                hkBarycentricVertexInterpolator_loadValue(ele, srcValue, value);
                m_objectToWorld.transformDirection(value, value);
                value.normalizeIfNotZero<3>();
                // Store the value
                hkBarycentricVertexInterpolator_storeValue(ele, value, dstBuffer.m_start);
                break;
            }
            case hkVertexFormat::USAGE_NORMAL:
            case hkVertexFormat::USAGE_BINORMAL:
            {
                hkVector4 value;
                // Load the value
                hkBarycentricVertexInterpolator_loadValue(ele, srcValue, value);
                m_objectToWorld.transformDirection(value, value);
                // We need to normalize because the transform may scale
                value.normalizeIfNotZero<3>();

                if (m_isInverted)
                {
                    // Flip the direction
                    value.setNeg<4>(value);
                }
                // Store the value
                hkBarycentricVertexInterpolator_storeValue(ele, value, dstBuffer.m_start);
                break;
            }
        }
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
