/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>

#include <Common/Base/Monitor/hkMonitorStream.h>

// this
#include <Common/GeometryUtilities/Mesh/Deform/hkSimdSkinningDeformer.h>

HK_FORCE_INLINE hkUint16 getIndex_Unindexed( const void*, int w )
{
	return (hkUint16)w;
}

HK_FORCE_INLINE hkUint16 getIndex_UInt8( const hkUint8* indices, int w )
{
	return indices[w];
}

HK_FORCE_INLINE hkUint16 getIndex_UInt16( const hkUint16* indices, int w )
{
	return indices[w];
}

template<typename IndexFunc, typename IndexType>
void deformInternal( const hkMatrix4* worldCompositeMatrices, const hkSimdSkinningDeformer::Binding& binding, IndexFunc F, const IndexType* indices )
{
	HK_TIMER_BEGIN("hkaSimdSkinningDeformer::DeformAlignedInput", this);


	hkFloat32* posOut = HK_NULL;
	int posOutStride = 0;
	hkFloat32* normOut = HK_NULL;
	int normOutStride = 0;
	hkFloat32* binormOut = HK_NULL;
	int binormOutStride = 0;
	hkFloat32* tangentOut = HK_NULL;
	int tangentOutStride = 0;

	const hkVector4* posIn = HK_NULL;
	int posInStride = 0;
	const hkVector4* normIn = HK_NULL;
	int normInStride = 0;
	const hkVector4* binormIn = HK_NULL;
	int binormInStride = 0;
	const hkVector4* tangentIn = HK_NULL;
	int tangentInStride = 0;

	const hkMeshVertexBuffer::LockedVertices& input = binding.m_input;
	const hkMeshVertexBuffer::LockedVertices& output = binding.m_output;

	int index;
	if ((index = input.findBufferIndex(hkVertexFormat::USAGE_POSITION, 0)) >= 0)
	{
		HK_ASSERT(0x2528211,output.m_buffers[index].m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32);
		posOut = (float*)output.m_buffers[index].m_start;
		posOutStride = output.m_buffers[index].m_stride / sizeof(hkFloat32);

		HK_ASSERT(0x2528211,input.m_buffers[index].m_element.m_dataType == hkVertexFormat::TYPE_VECTOR4);
		posIn = (const hkVector4*)input.m_buffers[index].m_start;
		posInStride = input.m_buffers[index].m_stride / sizeof(hkVector4);
	}

	if ((index = input.findBufferIndex(hkVertexFormat::USAGE_NORMAL, 0)) >= 0)
	{
		HK_ASSERT(0x2528211,output.m_buffers[index].m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32);
		normOut = (hkFloat32*)output.m_buffers[index].m_start;
		normOutStride = output.m_buffers[index].m_stride / sizeof(hkFloat32);

		HK_ASSERT(0x2528211,input.m_buffers[index].m_element.m_dataType == hkVertexFormat::TYPE_VECTOR4);
		normIn = (const hkVector4*)input.m_buffers[index].m_start;
		normInStride = input.m_buffers[index].m_stride / sizeof(hkVector4);
	}

	if ((index = input.findBufferIndex(hkVertexFormat::USAGE_TANGENT, 0)) >= 0)
	{
		HK_ASSERT(0x2528211,output.m_buffers[index].m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32);
		tangentOut = (hkFloat32*)output.m_buffers[index].m_start;
		tangentOutStride = output.m_buffers[index].m_stride / sizeof(hkFloat32);

		HK_ASSERT(0x2528211,input.m_buffers[index].m_element.m_dataType == hkVertexFormat::TYPE_VECTOR4);
		tangentIn = (const hkVector4*)input.m_buffers[index].m_start;
		tangentInStride = input.m_buffers[index].m_stride / sizeof(hkVector4);
	}

	if ((index = input.findBufferIndex(hkVertexFormat::USAGE_BINORMAL, 0)) >= 0)
	{
		HK_ASSERT(0x2528211,output.m_buffers[index].m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32);
		binormOut = (hkFloat32*)output.m_buffers[index].m_start;
		binormOutStride = output.m_buffers[index].m_stride / sizeof(hkFloat32);

		HK_ASSERT(0x2528211,input.m_buffers[index].m_element.m_dataType == hkVertexFormat::TYPE_VECTOR4);
		binormIn = (const hkVector4*)input.m_buffers[index].m_start;
		binormInStride = input.m_buffers[index].m_stride / sizeof(hkVector4);
	}

	const hkUint8* weights = binding.m_weights;
	const int weightsStride = binding.m_weightsStride;

	// Use the math libs vector4s.
	// If it is using SIMD, then this function will be faster
	// that using float macros. If the math lib is non simd
	// then this function should not copy the float vals, instead
	// it should use some simple float macros --> TODO
	hkVector4 resultP;
	hkVector4 resultN;
	hkVector4 resultB;
	hkVector4 resultT;

	bool normalizeDirections = binding.m_normalizeDirections;

	hkVector4 tVec;

	const hkSimdReal oneOver255 = hkSimdReal_Inv_255;

	const int numWeights = binding.m_numWeights;
	const int numVertices = binding.m_numVerts;

	for(int v = 0; v < numVertices; v++)
	{
		resultP.setZero();
		resultN.setZero();
		resultB.setZero();
		resultT.setZero();

		for (int w = 0; w < numWeights; w++)
		{
			if (weights[w] > 0) // worth checking for.
			{
				hkSimdReal sw; sw.setFromUint8(weights[w]);
				const hkSimdReal normalizedWeight = sw * oneOver255;
				hkUint16 transformIndex = F(indices,w);
				const hkMatrix4& t = worldCompositeMatrices[ transformIndex ];
				if (posIn)
				{
					t.transformPosition(*posIn, tVec);
					resultP.addMul(normalizedWeight, tVec);
				}
				if (normIn)
				{
					t.transformDirection(*normIn, tVec);
					resultN.addMul(normalizedWeight, tVec);
				}
				if (binormIn)
				{
					t.transformDirection(*binormIn, tVec);
					resultB.addMul(normalizedWeight, tVec);
				}
				if (tangentIn)
				{
					t.transformDirection(*tangentIn, tVec);
					resultT.addMul(normalizedWeight, tVec);
				}
			}
		}

		if (posIn)
		{
			resultP.store<3,HK_IO_NATIVE_ALIGNED>(posOut);
			posOut +=  posOutStride;
			posIn +=  posInStride;
		}
		if (normIn)
		{
			if (normalizeDirections)
			{
				resultN.normalizeIfNotZero<3>();
			}
			resultN.store<3,HK_IO_NATIVE_ALIGNED>(normOut);
			normOut += normOutStride;
			normIn += normInStride;
		}
		if (binormIn)
		{
			if (normalizeDirections)
			{
				resultB.normalizeIfNotZero<3>();
			}
			resultB.store<3,HK_IO_NATIVE_ALIGNED>(binormOut);
			binormOut += binormOutStride;
			binormIn += binormInStride;
		}
		if (tangentIn)
		{
			if (normalizeDirections)
			{
				resultT.normalizeIfNotZero<3>();
			}
			resultT.store<3,HK_IO_NATIVE_ALIGNED>(tangentOut);
			tangentOut += tangentOutStride;
			tangentIn += tangentInStride;
		}

		// Next
		weights += weightsStride;
		indices = hkAddByteOffsetConst( indices, binding.m_transformIndicesStride );
	}

	HK_TIMER_END();
}


void HK_CALL hkSimdSkinningDeformer::deform( const hkMatrix4* worldCompositeMatrices, const Binding& binding  )
{
	if ( binding.m_transformIndices8 != HK_NULL )
	{
		deformInternal( worldCompositeMatrices, binding, getIndex_UInt8, binding.m_transformIndices8 );
	}
	else if ( binding.m_transformIndices16 != HK_NULL )
	{
		deformInternal( worldCompositeMatrices, binding, getIndex_UInt16, binding.m_transformIndices16);
	}
	else
	{
		deformInternal( worldCompositeMatrices, binding, getIndex_Unindexed, (void*)HK_NULL );
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
