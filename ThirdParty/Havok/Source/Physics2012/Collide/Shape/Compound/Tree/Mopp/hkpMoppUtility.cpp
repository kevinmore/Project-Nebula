/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Internal/Collide/Mopp/Builder/hkbuilder.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppCompilerInput.h>

hkpMoppCode* HK_CALL hkpMoppUtility::buildCode(const hkpShapeContainer* shapeContainer, const hkpMoppCompilerInput& moppInput, hkArray<hkpMoppCodeReindexedTerminal>* reindexInfo)
{
	HK_CHECK_FLUSH_DENORMALS();

	hkpMoppCode* code;
	if ( moppInput.m_cachePrimitiveExtents )
	{
		hkpMoppCachedShapeMediator mediator( shapeContainer );
		code = hkpMoppUtility::buildCodeInternal(mediator, shapeContainer, moppInput, reindexInfo);
	}
	else
	{
		hkpMoppShapeMediator mediator( shapeContainer );
		code = hkpMoppUtility::buildCodeInternal(mediator, shapeContainer, moppInput, reindexInfo);
	}

	if ( code != HK_NULL && moppInput.m_enableChunkSubdivision )
	{
		code->m_buildType = hkpMoppCode::BUILT_WITH_CHUNK_SUBDIVISION;
	}
	else if( code != HK_NULL )
	{
		code->m_buildType = hkpMoppCode::BUILT_WITHOUT_CHUNK_SUBDIVISION;
	}

	return code;
}


hkpMoppCode* HK_CALL hkpMoppUtility::buildCodeInternal(hkpMoppMediator& mediator, const hkpShapeContainer* shapeContainer, const hkpMoppCompilerInput& moppInput, hkArray<hkpMoppCodeReindexedTerminal>* reindexInfo)
{
	HK_WARN(0x6e8d163b, "Building MOPP code at runtime can be slow. MOPP code \n" \
		"is a platform independent byte code. It can be preprocessed \n" \
		"and saved on PC and loaded on the required platform at runtime. \n");

	hkpMoppCompiler compiler;
#if defined(HK_PLATFORM_HAS_SPU)
	HK_ON_DEBUG( if (moppInput.m_enableChunkSubdivision != true) HK_WARN(0xf3545676, "On PS3 you have to set hkpMoppCompilerInput::m_enableChunkSubdivision to true to allow the spu to process the MOPP" ); );
#endif
	//
	// set up user scaling struct for cost functions
	//
	{
		hkpMoppCostFunction::hkpMoppSplitCostParams costParams;
		costParams.m_weightPrimitiveSplit = 1.0f;
		if ( moppInput.m_useShapeKeys == false || moppInput.m_enableChunkSubdivision )
		{
			costParams.m_weightPrimitiveIdSpread  = 0.0f;
		}
		compiler.setCostParams( costParams );
	}

	bool enableInterleavedBuilding = moppInput.m_enableInterleavedBuilding;
	if ( moppInput.m_enableChunkSubdivision )
	{	
		// chunk creation does not work with interleaved building
		enableInterleavedBuilding = false;
	}

	//
	// optionally control the assembler
	//
	{
		hkpMoppAssembler::hkpMoppAssemblerParams ap;

		ap.m_relativeFitToleranceOfInternalNodes = moppInput.getRelativeFitToleranceOfInternalNodes();
		ap.m_absoluteFitToleranceOfInternalNodes = moppInput.getAbsoluteFitToleranceOfInternalNodes();
		ap.m_absoluteFitToleranceOfTriangles = moppInput.getAbsoluteFitToleranceOfTriangles();
		ap.m_absoluteFitToleranceOfAxisAlignedTriangles = moppInput.getAbsoluteFitToleranceOfAxisAlignedTriangles();
		ap.m_interleavedBuildingEnabled = enableInterleavedBuilding;

		compiler.setAssemblerParams( ap );
	}

	{
		hkpMoppSplitter::hkpMoppSplitParams splitParams( HK_MOPP_MT_LANDSCAPE );
		if ( moppInput.m_enablePrimitiveSplitting)
		{
			splitParams.m_maxPrimitiveSplitsPerNode = 50;
		}
		else
		{
			splitParams.m_maxPrimitiveSplits = 0;
			splitParams.m_maxPrimitiveSplitsPerNode = 0;
		}
		splitParams.m_minRangeMaxListCheck = 5;
		splitParams.m_interleavedBuildingEnabled = enableInterleavedBuilding;

		compiler.setSplitParams( splitParams );
	}


	int bufferSize = compiler.calculateRequiredBufferSize(&mediator);
	char *buffer = hkMemTempBlockAlloc<char>(bufferSize);

	hkpMoppCompilerChunkInfo chunkInfo( HK_MOPP_CHUNK_SIZE ); 

	if ( moppInput.m_enableChunkSubdivision )
	{
		compiler.m_chunkInfo = &chunkInfo;
	}

	//This is where the built code is assigned to the hkGeometry
	hkpMoppCode *code = compiler.compile(&mediator, buffer, bufferSize);

	// Copy reindex info if required
	if (reindexInfo)
	{
		reindexInfo->insertAt(0, chunkInfo.m_reindexInfo.begin(), chunkInfo.m_reindexInfo.getSize() );
	}

	hkMemTempBlockFree( buffer, bufferSize );
	return code;
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
