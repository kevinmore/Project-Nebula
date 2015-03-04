/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//
// Havok Memory Optimized Partial Polytope Compiler
// This class generates the tree to be passed to the assembler.
// All primitives need to be added to the system and a filename specified BEFORE compilation.
//

#ifndef HK_COLLIDE2_MOPP_COMPILER_H
#define HK_COLLIDE2_MOPP_COMPILER_H

#include <Physics2012/Internal/Collide/Mopp/Builder/Splitter/hkpMoppSplitter.h>
#include <Physics2012/Internal/Collide/Mopp/Builder/Splitter/hkpMoppCostFunction.h>
#include <Physics2012/Internal/Collide/Mopp/Builder/Assembler/hkpMoppAssembler.h>

	// maximum number of nodes used if interleaved building is enabled
#define HK_MOPP_ENABLED_INTERLEAVED_BUILDING_SPLITTER_MAX_NODES 4096


struct hkpMoppCompilerChunkInfo
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppCompilerChunkInfo );

	hkpMoppCompilerChunkInfo( int maxChunkSize ): m_maxChunkSize(maxChunkSize), m_compressor(HK_NULL) { m_safetySize = 20; }

	/// The maximum size for a chunk
	int m_maxChunkSize;

	/// Access to each of the individual chunks
	struct Chunk
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppCompilerChunkInfo::Chunk );

		class hkpMoppCodeGenerator* m_code;
		int m_codeSize;
	};

	hkArray<hkpMoppCodeReindexedTerminal> m_reindexInfo;
	hkArray<Chunk>   m_chunks;

	/// A handle to the interface used to embed primitives into the MOPP
	class hkpPrimitiveCompressor* m_compressor;

	// An extra size, which is added. The reason is that it is not possible to
	// properly predict the correct size of a subtree
	int m_safetySize;

};


/// the MOPP compiler is the wrapper around different algorithms needed
/// to compile a set of convex primitives (defined by the mediator)
/// into a MOPP byte code.
class hkpMoppCompiler 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppCompiler );

		/// standard constructor
		hkpMoppCompiler( hkpMoppMeshType meshType = HK_MOPP_MT_LANDSCAPE );  

		/// standard destructor
		~hkpMoppCompiler();  

		/// optionally set splitter params
		/// for defining how the volume is split into a partial polytope hierarchy
		void setSplitParams( const hkpMoppSplitter::hkpMoppSplitParams& );

		/// optionally set some params
		void setCostParams( const hkpMoppCostFunction::hkpMoppSplitCostParams& );

		/// optionally set assembler params
		void setAssemblerParams  ( const hkpMoppAssembler::hkpMoppAssemblerParams& );

		/// optionally enable/disable interleaved building
		///
		/// The MOPP compiler has two operating modes:
		/// - Interleaved building enabled:
		///   the hkpMoppCompiler grabs a fixed buffer size (roughly 2 megabytes) independent of the number of triangles
		///   (which is good, if you have lots and lots of triangles).
		/// - Interleaved building disabled:
		///   it calculates the correct buffer size. However this buffer size can be extremely
		///   huge as each triangle takes more than 430 bytes.
		///
		/// By default interleaved building is enabled.
		void enableInterleavedBuilding(bool);

		/// Get the size of the temporary storage which is internally required by the compiler.
		/// Note: the compiler is doing very little allocations and deallocations except for generating the code.
		int calculateRequiredBufferSize( hkpMoppMediator* );

		/// compile the primitives defined by the mediator into the MOPP byte code.
		hkpMoppCode* compile(hkpMoppMediator* m_mediator, char* buffer = HK_NULL, int bufferSize = 0 );

	public:
		/// the root node of the compilation process.
		hkpMoppTreeNode* m_debugRootNode;

		struct hkpMoppCompilerChunkInfo* m_chunkInfo;

	protected:

		hkpMoppSplitter::hkpMoppSplitParams				m_splitParams;
		hkpMoppCostFunction::hkpMoppSplitCostParams		m_splitCostParams;
		hkpMoppAssembler::hkpMoppAssemblerParams		m_assemblerParams;
};

#endif // HK_COLLIDE2_MOPP_COMPILER_H

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
