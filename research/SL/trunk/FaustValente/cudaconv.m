% -*- Mode: Octave -*-

function [lat, avgLoadLat, avgStoreLat, avgCompLat, avgSetupCompLat, avgEmbedMemLat, avgGlbSync] = cudaconv_im(blockEdge, pyramidHeight, bench, DataEdge)
    %blockEdge = x(1); % the more it is, the more pyramid height benefits
    %pyramidHeight = x(2);
    
    %bench = 'hotspot';
    %bench = 'poisson';
    %bench = 'cell';
    %bench = 'dynproc';
    % input invariants per app
    if strcmp(bench, 'hotspot')
        % [#totalWarpInstsPerMP(bench) * #MP / #benchElement]
        %instTotalPerWarp = 3.62;  %4.3 %260078; %7963870; % the higher it is, the shorter the pyramid
        %instSetupPerWarp = 1.62; %1.002 % the number of instruction prior to the pyramid computation
        instTotalPerWarp = 7.08; %59039
        instSetupPerWarp = 3.14; %26139
        bankconflict = 1.0; % 105 for total, 0 for setup
        dataEdge = 500; % doesn't change the pyramidheight, the benefit is pushed lower a little bit if increase to 50000
        dimension =  2; % the lower it is, the more the pyramid
        halo = 2; % along one dimension. the higher it is, the lower the pyramid. why??
        loadFactor = 2;
        embededMem = 0; % the number of global load that is embeded in the pyramid computation
    elseif strcmp(bench, 'poisson')
        %instTotalPerWarp = 3.036;
        %instSetupPerWarp = 1.539;
        instTotalPerWarp = 5.73; %47740
        instSetupPerWarp = 3.02; %25142
        bankconflict = 1.0; %100 for total, 0 for setup
        dataEdge = 500;
        dimension = 2;
        halo = 2; % along one dimension. the higher it is, the lower the pyramid
        loadFactor = 1;
        embededMem = 0;
    elseif strcmp(bench, 'dynproc')
        %instTotalPerWarp = 1.16; %1.36
        %instSetupPerWarp = 0.5995; %0.368;
        instTotalPerWarp = 2.28; %7602
        instSetupPerWarp = 1.08; %3606
        bankconflict = 1.0; % 0 for total and 0 for setup
        dataEdge = 1000000;
        dimension = 1;
        halo = 2; % along one dimension. the higher it is, the lower the pyramid
        loadFactor = 1;
        embededMem = 1;
    elseif strcmp(bench, 'cell')
        %instTotalPerWarp = 41.573; %25.5;
        %instSetupPerWarp = 9.945; %12.923;
        instTotalPerWarp = 35.07; %252523 * 30 / 216000
        instSetupPerWarp = 16.66; %119955
        bankconflict = 2.0; %272000 for total, 17800 for setup
        dataEdge = 60;
        dimension = 3;
        halo = 2; % along one dimension. the higher it is, the lower the pyramid
        loadFactor = 1;
        embededMem = 0;
    else
        error('no such benchmark');
    end
    %halo = [1,2,3];
    
    dataEdge = DataEdge;
    numBlocks = (dataEdge./(blockEdge-(pyramidHeight).*halo)).^dimension;
    % input constants
    alpha = 1; % weight adjustment to compLat, default is 1
    CPUclock=3200000000;
    
    %device = 0;
    device = 1;
    if device == 0
        GPUclock=1460000000;
        % the lower it is, the shorter the pyramid
        glbSync = 95847.*CPUclock./GPUclock; % the more it is, the more important the glbSync reduction is, it increases the pyramid height
        % warpSize = 32; % accounted for in the instruction count
        numMPs = 4;       % the more it is, the more important the glbSync reduction is, so it increases the pyramid height
    else
        GPUclock=1300000000;
        % the lower it is, the shorter the pyramid
        % obtained by curve fitting using glbsync
        slant = 45.0; % ratio between theoretical result and experimental result
        %glbSync = (25.15*numBlocks+3937)./CPUclock.*GPUclock;
        %glbSync = (21.8258*numBlocks+16456)./CPUclock.*GPUclock;
        %glbSync=8246./CPUclock.*GPUclock; %3350
        glbSync=9033./CPUclock.*GPUclock;
        %glbSync=1950000;
        %glbSync = (29*numBlocks+8700)./CPUclock.*GPUclock;%./slant; % the more it is, the more important the glbSync reduction is, it increases the pyramid height
        % warpSize = 32; % accounted for in the instruction count
        numMPs = 30;       % the more it is, the more important the glbSync reduction is, so it increases the pyramid height
        bandwidth_BperCycle = 141.7*(2^30)/(1.3*(2^30));
        ReqSize = 4; % 512 Bytes ??? not 4 * coalesceWidth?
    end

    IPC =  0.5/bankconflict;       % the more it is, the lower compLat counts, move the minHeight closer to the bottom of loadlat
    % the following parameters determine the shape of avgLoadLat. 
    % the more the coefficient is, the less the curve is.
    coalesceWidth = 16;  % for hotspot, if 1, minheight = 3, if 8, minheight = 2
    uncontendedLat = 300;  % for hotspot, increase it to 30000 move the minheight from 2 to 4
    %cyclesPerMemReq = 4;   
    numBlocksPerMP = 8;
    %memQueueLat = uncontendedLat/channels/pipelineDepth; % reqs are divided in to 8 channels each with a pipeline of 4
    %memQueueLat = ReqSize/bandwidth_BperCycle/pipelineDepth*dimension;
    %factor = dimension^2.781;
    %factor = 2.78^dimension;
    factor = 5^(dimension-1);
    %factor = 1;
    memQueueLat = ReqSize*coalesceWidth/bandwidth_BperCycle*factor;
    
    % compute
    % instPerElementPerWarp = instTotalPerWarp./dataSize;
    numConcurrentBlocks = numBlocksPerMP*numMPs;
    %numConcurrentBlocks = numBlocks;
    
    loadLat = loadFactor.*numBlocks./numConcurrentBlocks.*memLat(workSet(blockEdge, dimension).*numConcurrentBlocks, ...
                coalesceWidth, memQueueLat, uncontendedLat);
    storeLat = numBlocks./numConcurrentBlocks.*memLat(workSet(blockEdge-(pyramidHeight)*halo, dimension).*numConcurrentBlocks, ...
                coalesceWidth, memQueueLat, uncontendedLat);
    computeLat = pyramidBlkCompLat(blockEdge, halo, dimension, pyramidHeight, ...
                IPC, ....
                instTotalPerWarp-instSetupPerWarp);
    setupCompLat = blkCompLat(workSet(blockEdge, dimension), ...
                   IPC, ...
                   instSetupPerWarp);
    embededMemLat = embededMem.*numBlocks./numConcurrentBlocks.*pyramidMemLat(blockEdge, numConcurrentBlocks, halo, dimension, pyramidHeight, ...
                   coalesceWidth, memQueueLat, uncontendedLat);
    
    avgLoadLat = loadLat./pyramidHeight; % going down then suddenly high [Category A, major]
    avgStoreLat = storeLat./pyramidHeight; % going down then suddenly high [A, minor]
    avgCompLat = computeLat.*numBlocks./pyramidHeight./numMPs; % going higher always [Category B, major]
    avgSetupCompLat = setupCompLat.*numBlocks./pyramidHeight./numMPs; % going down then suddenly high [A, negligible]
    avgEmbedMemLat = embededMemLat./pyramidHeight; % going higher always [B, minor]
    avgGlbSync = glbSync./pyramidHeight;
    
    lat = (glbSync + numBlocks./numMPs .*...
        (computeLat + setupCompLat) +  ...
        (loadLat + storeLat + embededMemLat) )...
         ./pyramidHeight;
     
    %lat = lat.*GPUclock./CPUclock;
                
    
    function ret = workSet(edge, dimension)
        ret = edge.^dimension;
    end
    
    function ret = memLat_old(numElements, coalesceWidth, channels, pipelineDepth, cyclesPerMemReq, uncontendedLat)
        % compute latency for load/store numElement elements from or to the
        % global memory
        ret = numElements./coalesceWidth./channels./pipelineDepth.*(cyclesPerMemReq+uncontendedLat);
    end


    function ret = memLat(numElements, coalesceWidth, memQueueLat, uncontendedLat)
        % compute latency for load/store numElement elements from or to the
        % global memory
        %warpSize = 32;
        %concurrentReqs = numMPs.*(warpSize./coalesceWidth).*(blockEdge^dimension);
        %rounds = numElements./coalesceWidth./concurrentReqs;
        concurrentReqs = numElements./coalesceWidth;
        ret = concurrentReqs.*memQueueLat+uncontendedLat;
        
        % bandwidth = 141.7*(2^30); %Bytes/sec
        % BytesPerCycle = bandwidth/GPUclock;
        % ret = concurrentReqs*4/BytesPerCycle;
    end
    
    function ret = pyramidMemLat_trzp(edge, halo, dimension, pyramidHeight, ...
                   coalesceWidth,  memQueueLat, uncontendedLat)
        ret = 0.0;
        for n = 1 : pyramidHeight
            set = workSet(edge-(n-1).*halo, dimension); % need more accurate estimate for diverged branches
            ret = ret + memLat(set, coalesceWidth,  memQueueLat, uncontendedLat);
        end
    end

    function ret = pyramidMemLat(edge, numBlocks, halo, dimension, pyramidHeight, ...
                    coalesceWidth, memQueueLat, uncontendedLat)
        set = workSet(edge-halo, dimension).*numBlocks; % need more accurate estimate for diverged branches
        ret = pyramidHeight.*memLat(set, coalesceWidth,  memQueueLat, uncontendedLat);
    end
               
    function ret = blkCompLat(numElements, IPC, instPerElementPerWarp)
        % compute latency for one block
        % assuming all data is in shared memory
        % and one block execute on one MP
        ret = instPerElementPerWarp./IPC.*numElements*alpha;
    end

    function ret = pyramidBlkCompLat_trpz(edge, halo, dimension, pyramidHeight, IPC, instPerElementPerWarp)
        % compute latency for one block
        % assuming all data is in shared memory
        % and one block execute on one MP
        ret = 0.0;
        for n = 1 : pyramidHeight
            set = workSet(edge-n.*halo, dimension); % need more accurate estimate for diverged branches
            ret = ret + blkCompLat(set, IPC, instPerElementPerWarp);
        end
    end

    function ret = pyramidBlkCompLat(edge, halo, dimension, pyramidHeight, IPC, instPerElementPerWarp)
        % compute latency for one block
        % assuming all data is in shared memory
        % and one block execute on one MP
        set = workSet(edge-halo, dimension); % need more accurate estimate for diverged branches
        ret = pyramidHeight*blkCompLat(set, IPC, instPerElementPerWarp);
    end
end
