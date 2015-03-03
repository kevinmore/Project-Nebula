/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SIMULATION_THREAD_CONTEXT_H
#define HKNP_SIMULATION_THREAD_CONTEXT_H

#include <Physics/Physics/hknpTypes.h>
#include <Common/Base/Container/CommandStream/hkCommandStream.h>
#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>
#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#include <Physics/Physics/Dynamics/Simulation/Utils/hknpSimulationDeterminismUtil.h>

struct hknpSpaceSplitterData;
class hknpDeactivationStepInfo;
class hknpCdPairStream;
class hknpShapeTagCodec;
class hknpSolverStepInfo;


/// A grid of command ranges.
/// This is the main mechanism for ensuring determinism in multi-threaded command dispatch.
class hknpCommandGrid : public hknpGrid<hkBlockStreamBase::LinkedRange>
{
	public:

		/// Dispatch all the commands in this grid.
		void dispatchCommands( hkPrimaryCommandDispatcher* dispatcher );
};


/// Helper struct which has convenient access to important variables for simulation.
/// There is one instance of this structure per thread.
/// Note that all data will be available on SPU if not mentioned otherwise.
class hknpSimulationThreadContext
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSimulationThreadContext );

		/// Constructor.
		hknpSimulationThreadContext();

		/// Setup.
		/// Note: we are not setting a writer for SPU threads yet
		void init( hknpWorld* world, hknpCommandGrid* commandGrid, hkBlockStreamAllocator* tempAllocator, int threadNumber, bool setupCommandStreamWriter );

		void appendCommands( hkBlockStream<hkCommand>* externalCommandStream );

		void dispatchCommands( hkPrimaryCommandDispatcher* dispatcher = HK_NULL );

		void shutdownThreadContext( hkThreadLocalBlockStreamAllocator* tempAllocator, hkThreadLocalBlockStreamAllocator* headAllocator );

		/// Clears the command stream and resets the command writer.
		void resetCommandStreamAndWriter( hkThreadLocalBlockStreamAllocator* tempAllocator, hkThreadLocalBlockStreamAllocator* heapAllocator );

		/// Finalize the command writer.
		HK_FORCE_INLINE void finalizeCommandWriters();

		/// Call to begin a batch of commands for the specified MT grid entry.
		HK_FORCE_INLINE void beginCommands( int gridEntryIndex ) const;

		/// Write a command.
		HK_FORCE_INLINE void execCommand( const hkCommand& command ) const;

#if !defined(HK_PLATFORM_SPU)
		/// Non virtual function of exec
		HK_FORCE_INLINE void appendCommand( const hkCommand& command ) const;
#endif

		/// Start writing a command of a given size. The command can then be initialized at will.
		HK_FORCE_INLINE hkCommand* beginCommand( int size ) const;

		/// Finish writing a command allocated using beginCommand. The command must be already finalized.
		HK_FORCE_INLINE void endCommand( const hkCommand* command ) const;

		/// Call at the end of a batch of commands for the specified MT grid entry.
		HK_FORCE_INLINE void endCommands( int gridEntryIndex ) const;

	public:

		//
		//	Global input variables
		//

		hkPadSpu<hknpWorld*>				m_world;			///< The world. Not available on SPU.
		hkPadSpu<const hknpBodyQuality*>	m_qualities;		///< All qualities
		hkPadSpu<hkUint32>					m_numQualities; 	///< Total number of qualities (for debug checks)
		hkPadSpu<const hknpMaterial*>		m_materials;		///< All materials
		hkPadSpu<hkUint32>					m_numMaterials; 	///< Total number of materials (for debug checks)
		hkPadSpu<hknpModifierManager*>  	m_modifierManager;	///< All registered modifiers
		hkPadSpu<const hknpShapeTagCodec*>	m_shapeTagCodec;	///< The codec used for decoding shape tags.

		//
		//	Collide specific
		//

		/// Embedded triangle shapes, pointed to by below m_triangleShapePrototypes.
		hknpInplaceTriangleShape m_triangleShapePrototype0;
		hknpInplaceTriangleShape m_triangleShapePrototype1;

		/// Triangles shape template, can be used by the collision detector to quickly convert 3/4 vertices into a triangle shape.
		/// Note that there is two of them to handle situations where both parent shapes yield runtime children (i.e. mesh vs mesh).
		hkPadSpu<hknpTriangleShape*> m_triangleShapePrototypes[2];

		//
		//	Solver specific (optional)
		//

		hkPadSpu<hknpDeactivationThreadData*> m_deactivationData;	///< optional output of the integrator
		hkPadSpu<hknpSolverStepInfo*>		  m_solverStepInfo;
		//
		//	Solver parallel tasks specific (optional)
		//

		hkPadSpu<hknpDeactivationStepInfo*> m_deactivationStepInfo;		///< optional step deactivation info
		hkPadSpu<hknpIdxRangeGrid*> m_cellIdxToGlobalSolverId;			///< optional cellIdxToGlobalSolverId map

		//
		//	Space splitter (optional)
		//

		hkPadSpu<hknpSpaceSplitterData*> m_spaceSplitterData;	///< optional output of the integrator

		//
		//	Internal variables
		//

		hkPadSpu<hkThreadLocalBlockStreamAllocator*> m_tempAllocator;	///< Temp allocator
		hkPadSpu<hkThreadLocalBlockStreamAllocator*> m_heapAllocator;

		//
		//	Command Related members
		//

		hkBlockStream<hkCommand>   m_commandBlockStream;	///< Stream storage of the commands.
		mutable hkPadSpu<hkBlockStreamCommandWriter*> m_commandWriter;	///< Command writer, used to sent events to the master thread.
		mutable hkPadSpu<hknpCommandGrid*>			m_commandGrid;		///< Command grid for determinism.
		mutable hkBlockStreamBase::LinkedRange		m_currentGridEntryRange;	///< Range tracking the commands to the current grid entry.

		mutable int m_currentGridEntryDebug;		///< Tracks the sanity of beginCommands/endCommands in debug mode.
};


#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.inl>

#endif // HKNP_SIMULATION_THREAD_CONTEXT_H

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
